"""Dense-only vs BM25 evaluation on the DSP dev queries.

Run: uv run python src/perplexity_from_scratch/evaluate_dense.py
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from dense_retrieval import dense_search
from evaluation import recall_at_k, reciprocal_rank_at_k
from inverted_index import build_inverted_index, search


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "evaluation"
TOP_K = 5
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def read_json(filename):
    return json.loads((DATA / filename).read_text(encoding="utf-8"))


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    documents = read_json("dsp_chunks.json")["documents"]
    queries = read_json("queries.json")
    qrels = read_json("qrels_proposed.json")
    dev_queries = [q for q in queries if q.get("split") == "dev"]

    if not dev_queries:
        raise ValueError("No dev queries found in queries.json")

    print(f"Corpus: {len(documents)} chunks | Dev queries: {len(dev_queries)}")
    index = build_inverted_index(documents)
    doc_map = {doc["id"]: doc for doc in documents}

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print("Encoding document chunks...")
    doc_embeddings = model.encode(
        [doc["text"] for doc in documents],
        batch_size=32,
        show_progress_bar=True,
    )
    assert len(doc_embeddings) == len(documents)

    records = []
    for q in dev_queries:
        query_id = q["query_id"]
        query = q["query"]
        relevant_ids = [
            doc_id for doc_id, grade in qrels[query_id].items() if grade >= 2
        ]

        bm25_ids = [
            item["id"]
            for item in search(query, index, doc_map, documents, topk=TOP_K)
        ]
        dense_ids = [
            item["id"]
            for item in dense_search(
                query, documents, doc_embeddings, model, topk=TOP_K
            )
        ]

        records.append({
            "query_id": query_id,
            "query": query,
            "relevant_ids": relevant_ids,
            "bm25_ids": bm25_ids,
            "dense_ids": dense_ids,
            "bm25_recall": recall_at_k(bm25_ids, relevant_ids, TOP_K),
            "dense_recall": recall_at_k(dense_ids, relevant_ids, TOP_K),
            "bm25_rr": reciprocal_rank_at_k(bm25_ids, relevant_ids, TOP_K),
            "dense_rr": reciprocal_rank_at_k(dense_ids, relevant_ids, TOP_K),
        })

    print("\n=== DEV SET EVALUATION ===")
    print(f"{'Retriever':<12} {'Recall@5':>12} {'MRR@5':>12}")
    for name, prefix in (("BM25", "bm25"), ("Dense", "dense")):
        recall = mean([r[f"{prefix}_recall"] for r in records])
        mrr = mean([r[f"{prefix}_rr"] for r in records])
        print(f"{name:<12} {recall:>12.4f} {mrr:>12.4f}")

    print("\n=== PER-QUERY COMPARISON ===")
    for r in records:
        print(
            f"{r['query_id']} | "
            f"BM25 R={r['bm25_recall']:.2f} RR={r['bm25_rr']:.2f} | "
            f"Dense R={r['dense_recall']:.2f} RR={r['dense_rr']:.2f}"
        )

    output_path = DATA / "dense_only_dev_results.json"
    output_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()