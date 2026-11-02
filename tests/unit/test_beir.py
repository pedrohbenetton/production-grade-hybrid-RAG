import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from hybrid_rag.evaluation.beir import download_beir, load_beir

CORPUS = [
    {"_id": "d1", "title": "Cats", "text": "Cats purr when relaxed.", "metadata": {}},
    {"_id": "d2", "title": "", "text": "Rockets burn propellant.", "metadata": {}},
    {"_id": "d3", "title": "Empty", "text": "", "metadata": {}},
]
QUERIES = [
    {"_id": "q1", "text": "why do cats purr"},
    {"_id": "q2", "text": "how do rockets work"},
    {"_id": "q9", "text": "a train-only query"},
]
QRELS = "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t2\nq2\td1\t0\n"


def write_dataset(root: Path, name: str = "tiny") -> Path:
    base = root / name
    (base / "qrels").mkdir(parents=True)
    (base / "corpus.jsonl").write_text("\n".join(map(json.dumps, CORPUS)), encoding="utf-8")
    (base / "queries.jsonl").write_text("\n".join(map(json.dumps, QUERIES)), encoding="utf-8")
    (base / "qrels" / "test.tsv").write_text(QRELS, encoding="utf-8")
    return base


def make_zip(directory: Path, name: str = "tiny") -> Path:
    source = write_dataset(directory / "src", name)
    archive = directory / f"{name}.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in source.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(source.parent).as_posix())
    return archive


def test_load_reads_corpus_queries_and_qrels_of_the_split(tmp_path: Path) -> None:
    write_dataset(tmp_path)

    dataset = load_beir(tmp_path, "tiny")

    assert dataset.name == "tiny"
    assert set(dataset.corpus) == {"d1", "d2", "d3"}
    assert dataset.corpus["d1"].title == "Cats"
    assert dataset.queries == {"q1": "why do cats purr", "q2": "how do rockets work"}
    assert dataset.qrels == {"q1": {"d1": 1}, "q2": {"d2": 2, "d1": 0}}


def test_missing_split_is_a_clear_error(tmp_path: Path) -> None:
    write_dataset(tmp_path)
    with pytest.raises(FileNotFoundError, match="dev"):
        load_beir(tmp_path, "tiny", split="dev")


def test_limit_queries_keeps_the_first_n_in_a_stable_order(tmp_path: Path) -> None:
    write_dataset(tmp_path)
    limited = load_beir(tmp_path, "tiny").limit_queries(1)

    assert list(limited.queries) == ["q1"]
    assert list(limited.qrels) == ["q1"]
    assert len(limited.corpus) == 3  # the corpus is never shrunk: distractors matter


def test_download_extracts_verifies_and_reports_the_archive_hash(tmp_path: Path) -> None:
    archive = make_zip(tmp_path / "remote")
    dest = tmp_path / "data"

    path, digest = download_beir("tiny", dest, base_url=archive.parent.as_uri())

    assert path == dest / "tiny"
    assert digest == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert load_beir(dest, "tiny").queries


def test_download_is_skipped_when_the_dataset_is_already_present(tmp_path: Path) -> None:
    archive = make_zip(tmp_path / "remote")
    dest = tmp_path / "data"
    first = download_beir("tiny", dest, base_url=archive.parent.as_uri())

    archive.unlink()  # a second call must not touch the network (or the missing file)
    second = download_beir("tiny", dest, base_url=archive.parent.as_uri())

    assert second == first


def test_archives_cannot_write_outside_the_destination(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    with zipfile.ZipFile(remote / "evil.zip", "w") as zf:
        zf.writestr("evil/corpus.jsonl", "")
        zf.writestr("../escaped.txt", "pwned")
    dest = tmp_path / "data"

    with pytest.raises(ValueError, match="unsafe"):
        download_beir("evil", dest, base_url=remote.as_uri())

    assert not (tmp_path / "escaped.txt").exists()
    assert not (dest / "evil").exists()
