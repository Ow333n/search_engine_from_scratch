
"""Compare BM25, Dense and RRF Hybrid retrieval on the DSP dev set.

From the repository root:
    uv run python src/perplexity_from_scratch/evaluate_hybrid_rrf.py

No Cross-Encoder or LLM is used in this experiment.
Qrels are provisional; an unjudged chunk is not necessarily irrelevant.
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from dense_retrieval import dense_search
from evaluation import recall_at_k, reciprocal_rank_at_k
from hybrid_search import hybrid_search
from inverted_index import build_inverted_index, search


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "evaluation"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
CANDIDATE_K = 20


def read_json(filename):
    return json.loads((DATA / filename).read_text(encoding="utf-8"))


def mean(values):
    return sum(values) / len(values) if values else 0.0


def ids(results):
    return [item["id"] for item in results]


def score(retrieved, relevant):
    return {
        "recall": recall_at_k(retrieved, relevant, TOP_K),
        "rr": reciprocal_rank_at_k(retrieved, relevant, TOP_K),
    }


def main():
    documents = read_json("dsp_chunks.json")["documents"]
    queries = read_json("queries.json")
    qrels = read_json("qrels_proposed.json")
    dev_queries = [q for q in queries if q.get("split") == "dev"]

    if not dev_queries:
        raise ValueError("No dev queries found in queries.json")

    print(f"Corpus: {len(documents)} chunks | Dev queries: {len(dev_queries)}")
    print(f"Top-K: {TOP_K} | Each retriever supplies up to {CANDIDATE_K} RRF candidates")
    print(f"Loading {MODEL_NAME}")

    index = build_inverted_index(documents)
    doc_map = {doc["id"]: doc for doc in documents}

    model = SentenceTransformer(MODEL_NAME)
    doc_embeddings = model.encode(
        [doc["text"] for doc in documents],
        batch_size=32,
        show_progress_bar=True,
    )

    records = []

    for q in dev_queries:
        query_id, query = q["query_id"], q["query"]

        relevant = [
            doc_id
            for doc_id, grade in qrels[query_id].items()
            if grade >= 2
        ]

        # BM25 Only
        bm25_ids = ids(
            search(query, index, doc_map, documents, topk=TOP_K)
        )

        # Dense Only
        dense_ids = ids(
            dense_search(
                query,
                documents,
                doc_embeddings,
                model,
                topk=TOP_K,
            )
        )

        # Hybrid RRF without Cross-Encoder
        hybrid_ids = ids(
            hybrid_search(
                query,
                index,
                doc_map,
                documents,
                doc_embeddings,
                model,
                topk=TOP_K,
                candidate_k=CANDIDATE_K,
                reranker=None,
            )
        )

        rankings = {
            "bm25": bm25_ids,
            "dense": dense_ids,
            "hybrid_rrf": hybrid_ids,
        }

        records.append({
            "query_id": query_id,
            "query": query,
            "relevant_ids": relevant,
            "rankings": rankings,
            "metrics": {
                method: score(result_ids, relevant)
                for method, result_ids in rankings.items()
            },
        })

    print("\n=== DEV SET EVALUATION ===")
    print(f"{'Retriever':<14}{'Recall@5':>12}{'MRR@5':>12}")

    for method in ("bm25", "dense", "hybrid_rrf"):
        recalls = [
            r["metrics"][method]["recall"]
            for r in records
        ]

        rrs = [
            r["metrics"][method]["rr"]
            for r in records
        ]

        print(
            f"{method:<14}"
            f"{mean(recalls):>12.4f}"
            f"{mean(rrs):>12.4f}"
        )

    print("\n=== PER-QUERY COMPARISON ===")

    for r in records:
        m = r["metrics"]

        print(
            f"{r['query_id']} | "
            f"BM25 R={m['bm25']['recall']:.2f} "
            f"RR={m['bm25']['rr']:.2f} | "
            f"Dense R={m['dense']['recall']:.2f} "
            f"RR={m['dense']['rr']:.2f} | "
            f"RRF R={m['hybrid_rrf']['recall']:.2f} "
            f"RR={m['hybrid_rrf']['rr']:.2f}"
        )

    print("\n=== RRF CHANGES VS DENSE (METRICS ONLY) ===")

    changed = 0

    for r in records:
        old = r["metrics"]["dense"]
        new = r["metrics"]["hybrid_rrf"]

        if old != new:
            changed += 1

            print(
                f"{r['query_id']} | "
                f"delta Recall={new['recall'] - old['recall']:+.3f} "
                f"delta RR={new['rr'] - old['rr']:+.3f}"
            )

            print("  Relevant:", r["relevant_ids"])
            print("  Dense Top-5:", r["rankings"]["dense"])
            print("  RRF   Top-5:", r["rankings"]["hybrid_rrf"])

    if changed == 0:
        print("No per-query Recall@5 or RR@5 changes.")

    output = DATA / "hybrid_rrf_dev_results.json"

    output.write_text(
        json.dumps(
            records,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
