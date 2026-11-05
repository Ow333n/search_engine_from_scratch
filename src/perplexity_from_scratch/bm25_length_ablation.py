"""Reproduce BM25 length-normalisation experiments on the DSP dev set.

Usage:
    uv run python src/perplexity_from_scratch/bm25_length_ablation.py
    uv run python src/perplexity_from_scratch/bm25_length_ablation.py --case Q055

Q055 is a previously inspected TEST query: --case is for diagnosis only,
not for selecting parameters. Labels are incomplete/proposed.
"""

import argparse
from collections import Counter

from evaluate_bm25 import load_evaluation_data
from evaluation import recall_at_k, reciprocal_rank_at_k
from idf_practice import cal_avgdl, idf, tf_saturated, tokenizer
from inverted_index import build_inverted_index, retrieve_candidates


B_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
BASELINE_B = 0.75
TOPK = 5


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser(description="BM25 b-parameter dev-set ablation")
    parser.add_argument("--case", help="Optional query ID to inspect (diagnostic only)")
    args = parser.parse_args()

    documents, queries, qrels = load_evaluation_data()
    dev_queries = [q for q in queries if q.get("split") == "dev"]
    index = build_inverted_index(documents)

    term_counts = {
        doc["id"]: Counter(tokenizer(doc["text"])) for doc in documents
    }
    doc_lengths = {
        doc_id: sum(counts.values()) for doc_id, counts in term_counts.items()
    }
    avgdl = cal_avgdl(documents)
    if avgdl == 0:
        raise ValueError("Cannot evaluate an empty-token corpus")

    all_terms = {term for q in queries for term in tokenizer(q["query"])}
    idf_by_term = {term: idf(term, documents) for term in all_terms}

    def rank(query, b, topk=TOPK):
        terms = set(tokenizer(query))
        results = []
        for doc_id in retrieve_candidates(query, index):
            counts = term_counts[doc_id]
            score = sum(
                idf_by_term[term] * tf_saturated(
                    counts[term], avgdl, doc_lengths[doc_id], k=1.2, b=b
                )
                for term in terms if counts[term] > 0
            )
            if score > 0:
                results.append((doc_id, score))
        results.sort(key=lambda item: item[1], reverse=True)
        return [doc_id for doc_id, _ in results[:topk]]

    def scores(query_item, b):
        retrieved = rank(query_item["query"], b)
        relevant = [
            doc_id for doc_id, grade in qrels[query_item["query_id"]].items()
            if grade >= 2
        ]
        return (
            recall_at_k(retrieved, relevant, TOPK),
            reciprocal_rank_at_k(retrieved, relevant, TOPK),
        )

    print(f"Dev queries: {len(dev_queries)} | BM25 length normalisation ablation")
    print(f"{'b':<8}{'Recall@5':<14}{'MRR@5':<14}{'Changed vs 0.75':<20}")
    baseline = {q["query_id"]: scores(q, BASELINE_B) for q in dev_queries}
    for b in B_VALUES:
        per_query = {q["query_id"]: scores(q, b) for q in dev_queries}
        changed = sum(
            per_query[qid] != baseline[qid] for qid in per_query
        )
        print(
            f"{b:<8.2f}"
            f"{mean([v[0] for v in per_query.values()]):<14.4f}"
            f"{mean([v[1] for v in per_query.values()]):<14.4f}"
            f"{changed:<20}"
        )

    if args.case:
        selected = next((q for q in queries if q["query_id"] == args.case), None)
        if selected is None:
            parser.error(f"Unknown query ID: {args.case}")
        relevant = [
            doc_id for doc_id, grade in qrels[selected["query_id"]].items()
            if grade >= 2
        ]
        print(f"\nDiagnostic case {args.case} | split={selected.get('split')}")
        print(selected["query"])
        for b in B_VALUES:
            rankings = rank(selected["query"], b, topk=len(documents))
            positions = {
                doc_id: (rankings.index(doc_id) + 1 if doc_id in rankings else None)
                for doc_id in relevant
            }
            print(f"b={b:.2f} | relevant chunk ranks: {positions}")


if __name__ == "__main__":
    main()
