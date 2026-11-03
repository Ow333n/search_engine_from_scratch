"""Validate the draft DSP relevance dataset against exported dsp_chunks.json."""
import json
from pathlib import Path

BASE = Path("data/evaluation")
corpus = json.loads((BASE / "dsp_chunks.json").read_text(encoding="utf-8"))
queries = json.loads((BASE / "queries.json").read_text(encoding="utf-8"))
qrels = json.loads((BASE / "qrels_proposed.json").read_text(encoding="utf-8"))

documents = corpus["documents"]
doc_ids = [d["id"] for d in documents]
assert len(doc_ids) == len(set(doc_ids)), "Duplicate corpus chunk IDs"
assert corpus["chunk_size"] == 100 and corpus["overlap"] == 20
query_ids = [q["query_id"] for q in queries]
assert len(query_ids) == len(set(query_ids)), "Duplicate query IDs"
assert set(query_ids) == set(qrels), "Query/qrel keys do not match"
known = set(doc_ids)
for qid, labels in qrels.items():
    assert labels, f"No draft labels for {qid}"
    assert set(labels).issubset(known), f"Unknown corpus ID for {qid}"
    assert all(level in (0, 1, 2, 3) for level in labels.values())
    assert any(level >= 2 for level in labels.values())

print("PASS: Corpus IDs, queries, and proposed qrels are structurally consistent")
print("Queries:", len(queries))
print("Chunks:", len(documents))
print("Draft judgments:", sum(len(v) for v in qrels.values()))
print("NOTE: Labels are proposals; this does not validate their semantic correctness.")
