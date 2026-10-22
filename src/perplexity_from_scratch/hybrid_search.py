
from collections import defaultdict
from inverted_index import search, build_inverted_index
from dense_retrieval import dense_search


def rrf_fusion(bm25_results, dense_results, k=60, topk=40):
    scores = defaultdict(float)

    for results in [bm25_results, dense_results]:
        for rank, item in enumerate(results, start=1):
            doc_id = item["id"]
            scores[doc_id] += 1 / (k + rank)

    results = []

    for doc_id, score in scores.items():
        results.append({
            "id": doc_id,
            "score": score
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:topk]


def hybrid_search(query, index, doc_map, documents,
                  doc_embeddings, model, topk=3,
                  candidate_k=20, reranker=None):

    candidate_k = max(candidate_k, topk)

    # 第一阶段：关键词 + 语义检索
    bm25_results = search(
        query, index, doc_map, documents,
        topk=candidate_k
    )

    dense_results = dense_search(
        query, documents, doc_embeddings,
        model, topk=candidate_k
    )

    # 第二阶段：RRF 合并候选
    results = rrf_fusion(
        bm25_results,
        dense_results,
        k=60,
        topk=candidate_k * 2
    )

    # 第三阶段：Cross-Encoder 重新排序
    if reranker is not None and results:

        pairs = []

        for item in results:
            doc = doc_map[item["id"]]
            pairs.append((query, doc["text"]))

        scores = reranker.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False
        )

        for item, score in zip(results, scores):
            item["rerank_score"] = float(score)

        results.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

    return results[:topk]


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    documents = [
        {"id": "D1", "text": "How to bake a chocolate cake"},
        {"id": "D2", "text": "Automobile maintenance and repair guide"},
        {"id": "D3", "text": "Python programming tutorial"},
        {"id": "D4", "text": "How to change engine oil in a vehicle"}
    ]

    index = build_inverted_index(documents)
    doc_map = {doc["id"]: doc for doc in documents}
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts)

    results = hybrid_search(
        "car repair", index, doc_map,
        documents, embeddings, model
    )

    print(results)
