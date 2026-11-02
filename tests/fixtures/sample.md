# Hybrid Search Guide

Hybrid search combines lexical and semantic retrieval.

## Sparse Retrieval

BM25 ranks documents using term statistics.

- Term frequency saturates with k1.
- Length normalization is controlled by b.

## Dense Retrieval

Dense models embed text into vectors.

```python
def cosine(a, b):
    return a @ b
```

### Fusion

| Method | Needs calibration |
|---|---|
| RRF | No |
| Linear | Yes |
