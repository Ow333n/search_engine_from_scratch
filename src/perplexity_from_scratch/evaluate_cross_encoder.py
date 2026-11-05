
"""Evaluate Hybrid RRF vs RRF + Cross-Encoder on the DSP dev set."""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer, CrossEncoder

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

INSPECT_QUERIES = {"Q011", "Q026", "Q051"}


def read_json(filename):
    return json.loads(
        (DATA / filename).read_text(encoding="utf-8")
    )


def metrics(retrieved_ids, relevant_ids):
    return {
        "recall": recall_at_k(
            retrieved_ids, relevant_ids, TOP_K
        ),
        "rr": reciprocal_rank_at_k(
            retrieved_ids, relevant_ids, TOP_K
        ),
    }


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    documents = read_json("dsp_chunks.json")["documents"]
    queries = read_json("queries.json")
    qrels = read_json("qrels_proposed.json")

    dev_queries = [
        q for q in queries if q.get("split") == "dev"
    ]

    if not dev_queries:
        raise ValueError("No dev queries found")

    print(f"Corpus: {len(documents)} chunks")
    print(f"Dev queries: {len(dev_queries)}")

    # Build index
    index = build_inverted_index(documents)
    doc_map = {doc["id"]: doc for doc in documents}

    # Dense embedding model
    print("Loading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    embeddings = model.encode(
        [doc["text"] for doc in documents],
        batch_size=32,
        show_progress_bar=True,
    )

    # Cross-Encoder
    print("Loading Cross-Encoder...")

    reranker = CrossEncoder(
        RERANKER_MODEL,
        max_length=512,
    )

    records = []

    for q in dev_queries:
        query_id = q["query_id"]
        query = q["query"]

        relevant_ids = [
            doc_id
            for doc_id, grade in qrels[query_id].items()
            if grade >= 2
        ]

        # Stage 1: BM25 Top-20
        bm25_results = search(
            query,
            index,
            doc_map,
            documents,
            topk=CANDIDATE_K,
        )

        # Stage 1: Dense Top-20
        dense_results = dense_search(
            query,
            documents,
            embeddings,
            model,
            topk=CANDIDATE_K,
        )

        # Stage 2: RRF candidate pool
        pool = rrf_fusion(
            bm25_results,
            dense_results,
            k=RRF_K,
            topk=CANDIDATE_K * 2,
        )

        # Original RRF Top-5
        rrf_ids = [
            item["id"]
            for item in pool[:TOP_K]
        ]

        # Stage 3: Cross-Encoder scores
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

        # Reorder the SAME candidate pool
        reranked = sorted(
            zip(pool, scores),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )

        ce_ids = [
            item["id"]
            for item, _ in reranked[:TOP_K]
        ]

        # === Cross-Encoder Score Inspection ===

        if query_id in INSPECT_QUERIES:
            print(
                f"\n=== {query_id}: "
                f"CROSS-ENCODER ANALYSIS ==="
            )
            print("Query:", query)

            rrf_ranks = {
                item["id"]: rank
                for rank, item in enumerate(pool, start=1)
            }

            ce_ranks = {
                item["id"]: rank
                for rank, (item, _) in enumerate(
                    reranked, start=1
                )
            }

            score_map = {
                item["id"]: float(score)
                for item, score in reranked
            }

            print("\nCross-Encoder Top-10:")

            for rank, (item, ce_score) in enumerate(
                reranked[:10], start=1
            ):
                doc_id = item["id"]
                doc = doc_map[doc_id]

                grade = qrels[query_id].get(
                    doc_id, "UNJUDGED"
                )

                print(
                    f"\nCE Rank {rank} | "
                    f"{doc_id} | "
                    f"CE Score {float(ce_score):.4f} | "
                    f"RRF Rank {rrf_ranks[doc_id]} | "
                    f"Grade {grade}"
                )

                print(
                    "Source:",
                    doc["source"],
                    "| Page:",
                    doc["page"]
                )

                print("Text:", doc["text"][:350])

            print("\nRelevant Chunk Positions:")

            for doc_id in relevant_ids:
                ce_score = score_map.get(doc_id)

                if ce_score is None:
                    score_text = "N/A"
                else:
                    score_text = f"{ce_score:.4f}"

                print(
                    f"{doc_id} | "
                    f"RRF Rank "
                    f"{rrf_ranks.get(doc_id, 'N/A')} | "
                    f"CE Rank "
                    f"{ce_ranks.get(doc_id, 'N/A')} | "
                    f"CE Score {score_text}"
                )

        # Store results
        records.append({
            "query_id": query_id,
            "query": query,
            "relevant_ids": relevant_ids,
            "candidate_count": len(pool),
            "relevant_in_candidate_pool": len(
                set(relevant_ids)
                & {item["id"] for item in pool}
            ),
            "rrf_ids": rrf_ids,
            "cross_encoder_ids": ce_ids,
            "rrf": metrics(rrf_ids, relevant_ids),
            "cross_encoder": metrics(
                ce_ids, relevant_ids
            ),
        })

    # Verify old RRF baseline
    baseline_path = DATA / "hybrid_rrf_dev_results.json"

    if baseline_path.exists():
        old_records = {
            r["query_id"]: r
            for r in read_json(baseline_path.name)
        }

        mismatches = [
            r["query_id"]
            for r in records
            if r["query_id"] in old_records
            and r["rrf_ids"] != old_records[
                r["query_id"]
            ]["rankings"]["hybrid_rrf"]
        ]

        if mismatches:
            raise RuntimeError(
                "RRF baseline mismatch: "
                + ", ".join(mismatches)
            )

        print("Prior RRF consistency check: PASS")

    # Overall metrics
    print("\n=== DEV SET EVALUATION ===")

    print(
        f"{'Method':<21}"
        f"{'Recall@5':>12}"
        f"{'MRR@5':>12}"
    )

    for name, key in [
        ("Hybrid RRF", "rrf"),
        ("RRF + Cross-Encoder", "cross_encoder"),
    ]:
        recall = mean([
            r[key]["recall"] for r in records
        ])

        mrr = mean([
            r[key]["rr"] for r in records
        ])

        print(
            f"{name:<21}"
            f"{recall:>12.4f}"
            f"{mrr:>12.4f}"
        )

    # Per-query analysis
    print("\n=== PER-QUERY DELTAS ===")

    for r in records:
        delta_recall = (
            r["cross_encoder"]["recall"]
            - r["rrf"]["recall"]
        )

        delta_rr = (
            r["cross_encoder"]["rr"]
            - r["rrf"]["rr"]
        )

        print(
            f"{r['query_id']} | "
            f"Delta Recall={delta_recall:+.3f} | "
            f"Delta RR={delta_rr:+.3f} | "
            f"Candidates={r['candidate_count']} | "
            f"Relevant in pool="
            f"{r['relevant_in_candidate_pool']}/"
            f"{len(r['relevant_ids'])}"
        )

        if (
            delta_recall != 0
            or delta_rr != 0
            or r["query_id"] in {"Q021", "Q026"}
        ):
            print(
                "  Relevant:",
                r["relevant_ids"]
            )

            print(
                "  RRF Top-5:",
                r["rrf_ids"]
            )

            print(
                "  CE Top-5: ",
                r["cross_encoder_ids"]
            )

    # Save experiment
    output = (
        DATA / "hybrid_cross_encoder_dev_results.json"
    )

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
