"""BEIR-format datasets (corpus.jsonl, queries.jsonl, qrels/<split>.tsv)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


class BeirDoc(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str = ""
    text: str = ""


class BeirDataset(BaseModel):
    name: str
    corpus: dict[str, BeirDoc]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]

    def limit_queries(self, n: int) -> Self:
        """First ``n`` queries only. The corpus is never shrunk: distractors are the point."""
        keep = list(self.queries)[:n]
        return self.model_copy(
            update={
                "queries": {q: self.queries[q] for q in keep},
                "qrels": {q: self.qrels[q] for q in keep if q in self.qrels},
            }
        )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_beir(root: Path, name: str, split: str = "test") -> BeirDataset:
    base = root / name
    qrels_path = base / "qrels" / f"{split}.tsv"
    if not qrels_path.exists():
        raise FileNotFoundError(f"{name} has no {split!r} split: {qrels_path} not found")

    qrels: dict[str, dict[str, int]] = {}
    for line in qrels_path.read_text(encoding="utf-8").splitlines()[1:]:  # header row
        if not line.strip():
            continue
        query_id, doc_id, score = line.split("\t")
        qrels.setdefault(query_id, {})[doc_id] = int(score)

    corpus = {
        str(row["_id"]): BeirDoc(title=str(row.get("title", "")), text=str(row.get("text", "")))
        for row in _read_jsonl(base / "corpus.jsonl")
    }
    queries = {
        str(row["_id"]): str(row["text"])
        for row in _read_jsonl(base / "queries.jsonl")
        if str(row["_id"]) in qrels  # queries.jsonl also holds the other splits
    }
    return BeirDataset(name=name, corpus=corpus, queries=queries, qrels=qrels)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_beir(
    name: str, dest: Path, *, base_url: str = BEIR_BASE_URL
) -> tuple[Path, str | None]:
    """Fetch and unpack ``<base_url>/<name>.zip`` into ``dest/<name>``.

    Returns the dataset directory and the SHA-256 of the archive (recorded in the report
    so results can be tied to the exact data). A dataset that is already unpacked is
    left alone and no request is made.
    """
    target = dest / name
    archive = dest / f"{name}.zip"
    if (target / "corpus.jsonl").exists():
        return target, sha256_file(archive) if archive.exists() else None

    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{base_url}/{name}.zip", timeout=60) as response:
        archive.write_bytes(response.read())

    staging = dest / f".{name}.extracting"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        with zipfile.ZipFile(archive) as zf:
            root = staging.resolve()
            for member in zf.namelist():  # validate everything before writing anything
                if not (staging / member).resolve().is_relative_to(root):
                    raise ValueError(f"unsafe path in archive: {member!r}")
            zf.extractall(staging)
        extracted = staging / name
        if not (extracted / "corpus.jsonl").exists():
            raise ValueError(f"archive does not contain {name}/corpus.jsonl")
        os.replace(extracted, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target, sha256_file(archive)
