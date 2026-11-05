
"""Ablate Cross-Encoder reranking depth on the DSP dev set."""

import json
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer

from dense_retrieval import dense_search
from evaluation import recall_at_k, reciprocal_rank_at_k
from hybrid_search import rrf_fusion
from inverted_index import build_inverted_index, search


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "evaluation"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K = 5
CANDIDATE_K = 20
RRF_K = 60

DEPTHS = (5, 10, 20)
METHODS = ("rrf", "ce_5", "ce_10", "ce_20", "ce_full")


def load_json(filename):
    return json.loads(
        (DATA / filename).read_text(encoding="utf-8")
    )


def metrics(ids, relevant_ids):
    return {
        "recall": recall_at_k(ids, relevant_ids, TOP_K),
        "rr": reciprocal_rank_at_k(ids, relevant_ids, TOP_K),
    }


def average(values):
    return sum(values) / len(values) if values else 0.0


def main():
    documents = load_json("dsp_chunks.json")["documents"]
    queries = load_json("queries.json")
    qrels = load_json("qrels_proposed.json")

    dev_queries = [
        q for q in queries if q.get("split") == "dev"
    ]

    if not dev_queries:
        raise ValueError("No dev queries found")

    index = build_inverted_index(documents)
    doc_map = {d["id"]: d for d in documents}

    print(
        f"Corpus: {len(documents)} chunks | "
        f"Dev: {len(dev_queries)} queries"
    )

    # Load embedding model
    print("Loading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    embeddings = model.encode(
        [d["text"] for d in documents],
        batch_size=32,
        show_progress_bar=True,
    )

    # Load Cross-Encoder
    print("Loading Cross-Encoder...")

    reranker = CrossEncoder(
        RERANKER_MODEL,
        max_length=512,
    )

    records = []

    for q in dev_queries:
        query_id = q["query_id"]
        query = q["query"]

        relevant = [
            doc_id
            for doc_id, grade in qrels[query_id].items()
            if grade >= 2
        ]

        # Stage 1: BM25 Top-20
        bm25 = search(
            query,
            index,
            doc_map,
            documents,
            topk=CANDIDATE_K,
        )

        # Stage 1: Dense Top-20
        dense = dense_search(
            query,
            documents,
            embeddings,
            model,
            topk=CANDIDATE_K,
        )

        # Stage 2: RRF Fusion
        pool = rrf_fusion(
            bm25,
            dense,
            k=RRF_K,
            topk=CANDIDATE_K * 2,
        )

        pool_ids = [
            item["id"] for item in pool
        ]

        # Score all candidates once
        # Reuse the same scores for every depth
        pairs = [
            (query, doc_map[item["id"]]["text"])
            for item in pool
        ]

        scores = (
            reranker.predict(
                pairs,
                batch_size=16,
                show_progress_bar=False,
            )
            if pairs else []
        )

        scores = [float(s) for s in scores]

        if len(scores) != len(pool):
            raise RuntimeError(
                f"Score count mismatch for {query_id}"
            )

        # Original RRF Top-5
        rankings = {
            "rrf": pool_ids[:TOP_K]
        }

        # CE@5, CE@10, CE@20
        for depth in DEPTHS:
            reranked = sorted(
                zip(pool[:depth], scores[:depth]),
                key=lambda pair: pair[1],
                reverse=True,
            )

            rankings[f"ce_{depth}"] = [
                item["id"]
                for item, _ in reranked[:TOP_K]
            ]

        # CE Full Pool
        full_reranked = sorted(
            zip(pool, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )

        rankings["ce_full"] = [
            item["id"]
            for item, _ in full_reranked[:TOP_K]
        ]

        records.append({
            "query_id": query_id,
            "query": query,
            "relevant_ids": relevant,
            "candidate_count": len(pool),
            "relevant_in_pool": len(
                set(relevant) & set(pool_ids)
            ),
            "rankings": rankings,
            "metrics": {
                name: metrics(rankings[name], relevant)
                for name in METHODS
            },
        })

    # Check against previous Cross-Encoder experiment
    previous_path = (
        DATA / "hybrid_cross_encoder_dev_results.json"
    )

    if previous_path.exists():
        old = {
            r["query_id"]: r
            for r in load_json(previous_path.name)
        }

        for r in records:
            query_id = r["query_id"]

            if query_id not in old:
                raise RuntimeError(
                    f"Missing prior query: {query_id}"
                )

            expected = old[query_id]

            if (
                r["rankings"]["rrf"]
                != expected["rrf_ids"]
            ):
                raise RuntimeError(
                    f"RRF mismatch: {query_id}"
                )

            if (
                r["rankings"]["ce_full"]
                != expected["cross_encoder_ids"]
            ):
                raise RuntimeError(
                    f"Full CE mismatch: {query_id}"
                )

        print(
            "Previous RRF / Full CE "
            "consistency checks: PASS"
        )

    # Overall metrics
    print("\n=== RERANKING DEPTH ABLATION ===")

    print(
        f"{'Method':<12}"
        f"{'Recall@5':>10}"
        f"{'MRR@5':>10}"
        f"{'Delta R':>10}"
        f"{'Delta RR':>10}"
    )

    baseline_recall = average([
        r["metrics"]["rrf"]["recall"]
        for r in records
    ])

    baseline_mrr = average([
        r["metrics"]["rrf"]["rr"]
        for r in records
    ])

    for name in METHODS:
        recall = average([
            r["metrics"][name]["recall"]
            for r in records
        ])

        mrr = average([
            r["metrics"][name]["rr"]
            for r in records
        ])

        print(
            f"{name:<12}"
            f"{recall:>10.4f}"
            f"{mrr:>10.4f}"
            f"{recall - baseline_recall:>+10.4f}"
            f"{mrr - baseline_mrr:>+10.4f}"
        )

    # Per-query ranking comparison
    print("\n=== PER-QUERY RR@5 BY DEPTH ===")

    print(
        f"{'Query':<7}"
        f"{'RRF':>7}"
        f"{'CE@5':>7}"
        f"{'CE@10':>7}"
        f"{'CE@20':>7}"
        f"{'Full':>7}"
    )

    for r in records:
        m = r["metrics"]

        values = " ".join(
            f"{m[name]['rr']:>7.3f}"
            for name in METHODS
        )

        print(f"{r['query_id']:<7} {values}")

    # Inspect representative queries
    print("\n=== INSPECT Q011, Q021, Q026, Q051 ===")

    inspect_ids = {
        "Q011", "Q021", "Q026", "Q051"
    }

    for r in records:
        if r["query_id"] not in inspect_ids:
            continue

        print(
            f"\n{r['query_id']} | {r['query']}"
        )

        print(
            f"Pool: {r['candidate_count']} | "
            f"Relevant in pool: "
            f"{r['relevant_in_pool']}/"
            f"{len(r['relevant_ids'])}"
        )

        print("Relevant:", r["relevant_ids"])

        for name in METHODS:
            m = r["metrics"][name]

            print(
                f"{name:<8} "
                f"R={m['recall']:.3f} "
                f"RR={m['rr']:.3f} "
                f"Top-5={r['rankings'][name]}"
            )

    # Save evaluation results
    output = DATA / "rerank_depth_dev_results.json"

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
