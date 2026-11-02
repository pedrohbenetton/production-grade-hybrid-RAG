# Architecture — Production-Grade Hybrid RAG

> Status: **Spec v0.1** (design-first; implementation follows this document, test-first).

## 1. Design goals

| Goal | Concrete decision |
|---|---|
| Retrieval quality over the naive baseline | Dense + BM25 → RRF → cross-encoder rerank |
| Groundedness | Claim-level NLI verification before an answer is returned |
| Reproducibility | Deterministic ingestion (content-hash IDs), pinned models, benchmark on a public dataset with qrels |
| Runs anywhere | Every external dependency has a local fallback (Qdrant local mode, FastEmbed ONNX, FlashRank) → full test suite runs offline, no Docker required |
| Swappable components | Every stage is a `Protocol`; concrete adapters are injected via settings |

Non-goals (v1): multi-tenancy, auth, streaming UI, agentic multi-hop retrieval.

---

## 2. System overview

```mermaid
flowchart LR
    subgraph Ingestion["Ingestion (offline / POST /ingest)"]
        L[Loaders<br/>MD · PDF · HTML] --> N[Normalizer<br/>→ DocumentTree]
        N --> C[Chunker<br/>structural → semantic]
        C --> X[Contextualizer<br/>breadcrumb + optional LLM context]
        X --> E[Embedder<br/>dense]
        X --> T[Tokenizer<br/>BM25 sparse]
        E --> VS[(Qdrant<br/>dense collection)]
        T --> BM[(BM25 index)]
    end

    subgraph Query["Query path (POST /query)"]
        Q[Query] --> QP[Query processor]
        QP --> DR[Dense retriever<br/>top-k_d]
        QP --> SR[Sparse retriever<br/>top-k_s]
        DR --> F[RRF fusion]
        SR --> F
        F --> R[Reranker<br/>FlashRank / Cohere]
        R --> CB[Context builder<br/>token budget + citations]
        CB --> G[Generator LLM]
        G --> GR[Guardrails<br/>claim extraction + NLI]
        GR --> A[Answer + citations<br/>+ faithfulness score]
    end

    VS -.-> DR
    BM -.-> SR
```

---

## 3. Ingestion pipeline

### 3.1 Stages and contracts

Each stage is a pure transformation with a typed Pydantic v2 boundary:

```
bytes ──Loader──▶ RawDocument ──Normalizer──▶ DocumentTree ──Chunker──▶ list[Chunk]
      ──Contextualizer──▶ list[ContextualChunk] ──Indexer──▶ IndexReport
```

| Model | Key fields |
|---|---|
| `RawDocument` | `source_uri`, `mime_type`, `content: bytes`, `sha256` |
| `DocumentTree` | `doc_id`, `title`, `sections: list[Section]` (recursive: `heading`, `level`, `blocks`) |
| `Block` | `kind: Literal["paragraph","list","table","code"]`, `text`, `page: int \| None` |
| `Chunk` | `chunk_id`, `doc_id`, `text`, `section_path: tuple[str, ...]`, `token_count`, `position` |
| `ContextualChunk` | `Chunk` + `context_prefix`, `embed_text` (prefix + text) |

### 3.2 Loaders → one canonical tree

The key idea: **all formats are normalized into the same `DocumentTree`**, so chunking logic is format-agnostic.

| Format | Library | Structure recovery |
|---|---|---|
| Markdown | `markdown-it-py` | ATX/Setext headings → section levels; fenced code kept atomic |
| HTML | `selectolax` (lexbor) | Strip `nav/footer/script/style`; `h1–h6` → levels; `<table>` → Markdown table block |
| PDF | `pymupdf` | Font-size clustering per document → heading levels; header/footer removal by repeated-line detection across pages; page number kept in `Block.page` |

### 3.3 Chunking strategy (two passes)

**Pass 1 — Structural.** Walk the tree; each section is a candidate chunk. Code blocks and tables are never split.

**Pass 2 — Semantic split (only for sections > `max_tokens`).**
1. Sentence-split the section.
2. For each gap *i*, embed the `w+1` sentences ending at *i* and the `w+1` starting at *i+1* (TextTiling-style); cosine distance between the two. Gap-centred windows keep the peak on the topic shift — symmetric per-sentence windows were measured to shift it one sentence early.
3. Break where distance > percentile `p` (default 90th) of the distances in that section.
4. Merge resulting pieces smaller than `min_tokens` into their neighbour; enforce `max_tokens` hard cap with a recursive fallback.

Sections < `min_tokens` are merged with the next sibling under the same parent.

Defaults: `min_tokens=64`, `max_tokens=512`, `overlap=0` (semantic boundaries make overlap mostly redundant; configurable).

### 3.4 Contextualization

Isolated chunks lose referents ("it", "the method above"). Two levels:

1. **Breadcrumb (always, zero cost):** `"{title} > {section_path}"` prepended to `embed_text`.
2. **LLM context (opt-in, `CONTEXTUAL_LLM=true`):** a 50–100 token summary situating the chunk within the full document ("Contextual Retrieval", Anthropic 2024). Uses prompt caching on the document body to keep cost linear.

`embed_text` feeds **both** the dense embedder and BM25 — the prefix also improves lexical matching on section terms. The original `text` is what goes into the LLM prompt.

### 3.5 Idempotency

- `doc_id = sha256(source_uri)`, `chunk_id = sha256(doc_id ‖ section_path ‖ text ‖ occurrence)` (length-prefixed parts; `occurrence` disambiguates repeated text). Position is excluded so inserting a paragraph does not re-key every later chunk.
- Re-ingesting an unchanged document is a no-op; a changed document deletes stale chunk IDs (set difference) and upserts new ones.
- `IndexReport` returns `added / unchanged / removed` counts.

---

## 4. Retrieval engine

### 4.1 Dense retrieval
- Store: **Qdrant** (local on-disk mode for dev/tests, server mode via `QDRANT_URL` in prod).
- Embeddings: `BAAI/bge-small-en-v1.5` via FastEmbed (ONNX, CPU, 384-d). Swappable through `Embedder` protocol.
- Distance: cosine; HNSW defaults (`m=16`, `ef_construct=100`); keyword payload index on `doc_id` (server mode).
- Point ID = `uuid5(namespace, chunk_id)` — deterministic, so upserts are idempotent. Opening a collection with a different vector size fails fast (`DimensionMismatchError`) instead of silently mixing embedding spaces.
- Asymmetric embeddings: `embed_query` vs `embed` (passage) apply the model's query/passage prompts.

### 4.1.1 Indexing (`indexing.py`)
`ids_for_doc(doc_id)` from Qdrant → `diff_chunks` → embed **only** `to_add` (batched) → upsert dense + sparse → delete `to_remove`. Add-before-remove so a concurrent reader never sees a document vanish mid-update. An unchanged document costs zero embedding calls (tested).

### 4.2 Sparse retrieval
- **BM25 (Okapi)**, `k1=1.2`, `b=0.75`, Lucene IDF `ln(1 + (N−df+0.5)/(df+0.5))`, implemented in-house as an **incremental** inverted index (`retrieval/sparse.py`). Static libraries (`bm25s`, `rank_bm25`) must rebuild the whole corpus on every change; here N, df and avgdl are maintained on add/remove, so removing a chunk is provably equivalent to never indexing it (tested).
- Not persisted: Qdrant payloads store the full chunk, so Qdrant is the **system of record** and BM25 is rebuilt from it at startup (`rebuild_sparse`, O(total tokens)). Comfortable up to ~10⁵ chunks; beyond that, switch to Qdrant native sparse vectors with the IDF modifier behind the same interface.
- Analyzer: lowercase → Unicode NFKC → stopwords → Snowball stemmer. Same analyzer at index and query time (unit-tested invariant).

### 4.3 Reciprocal Rank Fusion

Score-scale-free fusion of `R` ranked lists:

$$\mathrm{RRF}(d) = \sum_{r \in R} \frac{w_r}{k + \mathrm{rank}_r(d)}$$

- `k = 60` (Cormack et al., 2009), per-retriever weights `w_r` (default 1.0).
- Why RRF instead of score interpolation: cosine similarity and BM25 scores live on incomparable, query-dependent scales; ranks don't need calibration.
- Candidates: `k_d = k_s = 50` → fused top `N = 30` → reranker.

### 4.4 Reranking
- Default: **FlashRank** (`ms-marco-MiniLM-L-12-v2`, local cross-encoder).
- Optional: **Cohere Rerank** when `COHERE_API_KEY` is set.
- Cross-encoders score `(query, chunk)` jointly → much higher precision than bi-encoders, but O(N) inference — hence applied only to the fused top-N.
- Output: top `k_final = 5`. The funnel never narrows below the requested `k` (candidates = `max(k, N)`).
- **Measured cost** (CPU, 30 candidates of real abstracts, max_length 512): MiniLM-L-12 p50 ≈3.6 s / p95 ≈4.7 s per query; TinyBERT-L-2 ≈157 ms; `hybrid` without rerank ≈24 ms. On BEIR/SciFact neither reranker improved ranking quality over plain hybrid (§7), so reranking is a per-domain decision to validate with the benchmark, not a default win. Generic MS MARCO cross-encoders are the likely mismatch for scientific claim retrieval (a hypothesis, not tested here).

### 4.5 Context builder
Packs reranked chunks into a token budget, ordered by score, each tagged `[n]` with `doc_id`, `section_path`, `page` for citations.

---

## 5. Generation & guardrails

```mermaid
sequenceDiagram
    participant API
    participant Gen as Generator
    participant G as Guardrails
    API->>Gen: question + numbered context
    Gen-->>API: draft answer with [n] citations
    API->>G: draft + context
    G->>G: split into atomic claims
    G->>G: NLI(claim, cited chunks) → entail / neutral / contradict
    G-->>API: faithfulness = supported / total, per-claim verdicts
    alt faithfulness < threshold
        API-->>API: drop unsupported claims or abstain
    end
```

Generator: **Claude** via the official `anthropic` SDK, behind an `LLMClient` protocol (fake in tests).

1. **Grounded prompt:** answer only from context, cite `[n]`, say "insufficient context" otherwise.
2. **Citation validity:** every `[n]` must reference an existing context chunk (deterministic check).
3. **Claim-level faithfulness:** sentences → claims (the prompt demands one claim per sentence, so sentence-level approximates atomic claims without a second LLM call); each verified by an NLI cross-encoder (`cross-encoder/nli-deberta-v3-small`, int8 ONNX on CPU — no torch) against the chunks it cites (fallback: all context). Passages are scored one by one, not concatenated (512-token window), so a claim that needs two passages combined counts as unsupported — a deliberate, conservative trade-off. Out-of-range `[n]` markers are reported in `invalid_citations` and stripped from filtered answers.
   **Measured:** true paraphrases score entailment ≥ 0.99, unsupported claims < 0.15, in both int8 and fp32. But SNLI/MNLI-style models label *off-topic* claims "contradiction" rather than "neutral", so decisions use **only** `entailed`; the neutral/contradicted split is informational.
   `faithfulness = |entailed claims| / |claims|`.
4. **Policy (configurable):** `annotate` (return verdicts) · `filter` (remove unsupported claims) · `abstain` (below threshold → refuse).
   `filter` rebuilds the answer from the entailed sentences only, so unsupported text never reaches the client; every claim (including dropped ones) is still returned in `claims[]`.
5. **Retrieval-side guard:** if there are no hits, or the top hit scores below `min_relevance`, skip generation and abstain — the cheapest hallucination is the one never generated. Scores are only comparable within a mode, so the threshold is meaningful for reranker scores, not RRF scores (off by default).
6. **Prompt-injection hygiene:** passages are wrapped in `<passage>` tags, escaped so they cannot close their own tag, and the system prompt declares them untrusted data. This lowers the risk; the NLI check is the actual backstop, since an injected instruction produces claims no passage entails.

Abstain reasons: `no_relevant_context` · `model_abstained` (exact refusal phrase) · `no_supported_claims` · `low_faithfulness`.

The response always exposes `faithfulness`, `claims[]`, and `citations[]`, so clients can enforce their own policy.

---

## 6. API (FastAPI)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/ingest` | Upload file(s) or URL; returns `IndexReport` |
| `POST` | `/v1/query` | Full RAG: retrieve → rerank → generate → verify |
| `POST` | `/v1/retrieve` | Retrieval only; `mode: dense \| sparse \| hybrid \| hybrid_rerank` — the same four systems the benchmark compares (`dense` = naive RAG) |
| `DELETE` | `/v1/documents/{doc_id}` | Remove a document and its chunks |
| `GET` | `/health` · `/ready` | Liveness / readiness (index loaded, models warm) |

OpenAPI/Swagger at `/docs`, every schema with examples. Per-stage latency returned in a `timings` field and as structured logs.

---

## 7. Evaluation & benchmark

- **Datasets:**
  1. BEIR **SciFact** (5,183 abstracts, 300 test claims, scientific domain).
  2. BEIR **NFCorpus** (3,633 PubMed articles, 323 natural language medical questions).
  Both downloaded from the official BEIR host with SHA-256 archive checksums recorded.
- **Same pipeline as production:** every document is ingested through parse → chunk → contextualize → index (not a shortcut path), and queries go through the same `HybridRetriever` the API uses. Rankings are scored per *document*: a document's rank is its best chunk's rank.
- **Systems compared** (same chunks, same embedder):
  1. Naive RAG — dense only
  2. BM25 only
  3. Hybrid — dense + BM25 + RRF
  4. Hybrid + rerank (MiniLM-L-12)
  5. Hybrid + rerank (TinyBERT-L-2)
- **Retrieval metrics:** P@1/5/10, R@10, MRR@10, nDCG@10 (`evaluation/metrics.py`, unit-tested against hand-computed values). Differences between systems are reported as **paired bootstrap 95% CIs over queries**: with ~300 queries, a gap of one or two points is often noise, and the report says so (✓/✗ only when the interval excludes 0).
- **Sanity anchors:** published BEIR nDCG@10 on SciFact is ≈0.665 for BM25 and ≈0.71 for bge-small; this repo measures 0.692 and 0.720 (in-house BM25; int8 ONNX bge-small with title prepended) — same ballpark, so the harness is not silently broken.
- **Generation metrics:** faithfulness is computed per request (§5); it is not part of this retrieval benchmark.
- **Reproduce:**
  ```bash
  uv run --group bench hybrid-rag-bench --dataset scifact   # outputs docs/benchmark.{json,md,png}
  uv run --group bench hybrid-rag-bench --dataset nfcorpus  # outputs docs/benchmark_nfcorpus.{json,md,png}
  ```
  `--report-only` regenerates the Markdown and chart from the saved JSON.
- **Results:**
  - **SciFact (see `docs/benchmark.md`):** Hybrid ranks first (+0.015 nDCG@10), but the gain over naive RAG is within statistical noise (CI [−0.012, +0.039]).
  - **NFCorpus (see `docs/benchmark_nfcorpus.md`):** Hybrid significantly outperforms Naive RAG across P@1 (+0.071, CI [+0.028, +0.115] ✓), MRR@10 (+0.050, CI [+0.021, +0.080] ✓), and nDCG@10 (+0.026, CI [+0.012, +0.040] ✓), demonstrating that hybrid retrieval excels when vocabulary mismatch between query and document is pronounced.
  - **Rerankers:** Across both corpora, generic MS MARCO cross-encoders failed to beat plain RRF hybrid fusion while introducing 15–300× higher latency.

---

## 8. Repository layout

```
src/hybrid_rag/
  config.py            # pydantic-settings
  domain/models.py     # Pydantic v2 schemas (§3.1)
  ingestion/{loaders,normalize,chunking,contextualize,pipeline}.py
  retrieval/{dense,sparse,fusion,rerank,hybrid}.py
  generation/{llm,prompts,context}.py
  guardrails/{claims,nli,policy}.py
  api/{app,routes,schemas,container}.py
  evaluation/{beir,metrics,stats,runner,report,cli}.py   # hybrid-rag-bench
tests/{unit,integration}/
```

Tooling: `uv`, `ruff`, `mypy --strict`, `pytest` (+ `pytest-asyncio`, coverage ≥ 85%), GitHub Actions CI, Dockerfile + `docker-compose` (API + Qdrant).

## 9. Testing strategy

| Layer | Examples |
|---|---|
| Unit | RRF math (ties, disjoint lists, weights); BM25 analyzer symmetry; chunker invariants (no chunk > max_tokens, code blocks intact, all text covered exactly once); metric correctness |
| Integration | Ingest fixture MD/PDF/HTML → Qdrant local → retrieve known answer; idempotent re-ingest; API contract via `httpx.AsyncClient` |
| Guardrails | Fake LLM returning a planted unsupported claim → must be flagged / filtered |

External APIs (LLM, Cohere) are behind protocols and replaced by fakes in tests — the suite runs offline.
