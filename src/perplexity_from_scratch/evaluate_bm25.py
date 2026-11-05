"""Evaluate the BM25-only retriever against the proposed DSP relevance labels.

Usage:
    uv run python src/perplexity_from_scratch/evaluate_bm25.py
    uv run python src/perplexity_from_scratch/evaluate_bm25.py --split dev
    uv run python src/perplexity_from_scratch/evaluate_bm25.py --worst 10

The qrels are incomplete/proposed, so reported scores are provisional.
"""

import argparse
import json
from pathlib import Path

from evaluation import recall_at_k, reciprocal_rank_at_k
from inverted_index import build_inverted_index, search


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "evaluation"


def load_evaluation_data():
    documents = json.loads((DATA / "dsp_chunks.json").read_text(encoding="utf-8"))["documents"]
    queries = json.loads((DATA / "queries.json").read_text(encoding="utf-8"))
    qrels = json.loads((DATA / "qrels_proposed.json").read_text(encoding="utf-8"))
    return documents, queries, qrels


def evaluate(queries, documents, qrels, topk=5):
    index = build_inverted_index(documents)
    doc_map = {doc["id"]: doc for doc in documents}
    records = []

    for item in queries:
        query_id = item["query_id"]
        retrieved = search(item["query"], index, doc_map, documents, topk=topk)
        retrieved_ids = [result["id"] for result in retrieved]
        relevant_ids = [
            doc_id for doc_id, grade in qrels[query_id].items() if grade >= 2
        ]

        records.append({
            "query_id": query_id,
            "query": item["query"],
            "split": item.get("split"),
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
            "recall": recall_at_k(retrieved_ids, relevant_ids, k=topk),
            "rr": reciprocal_rank_at_k(retrieved_ids, relevant_ids, k=topk),
        })

    return records


def print_summary(records, topk):
    if not records:
        print("No queries selected.")
        return

    recall = sum(record["recall"] for record in records) / len(records)
    mrr = sum(record["rr"] for record in records) / len(records)
    print(f"Queries evaluated: {len(records)}")
    print(f"Mean Recall@{topk}: {recall:.10f}")
    print(f"MRR@{topk}: {mrr:.10f}")


def main():
    parser = argparse.ArgumentParser(description="BM25-only DSP retrieval evaluation")
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--worst", type=int, default=0, help="Print N lowest-recall queries")
    args = parser.parse_args()

    if args.topk <= 0 or args.worst < 0:
        parser.error("--topk must be positive and --worst must be nonnegative")

    documents, queries, qrels = load_evaluation_data()
    if args.split != "all":
        queries = [item for item in queries if item.get("split") == args.split]

    records = evaluate(queries, documents, qrels, topk=args.topk)
    print(f"BM25 only | split={args.split} | corpus={len(documents)} chunks")
    print_summary(records, args.topk)

    if args.worst:
        print("\nLowest-recall queries:")
        for record in sorted(records, key=lambda row: (row["recall"], row["rr"]))[:args.worst]:
            print(f"{record['query_id']} | Recall={record['recall']:.3f} | RR={record['rr']:.3f}")
            print(f"  {record['query']}")
            print(f"  Retrieved: {record['retrieved_ids']}")
            print(f"  Relevant:  {record['relevant_ids']}")


if __name__ == "__main__":
    main()