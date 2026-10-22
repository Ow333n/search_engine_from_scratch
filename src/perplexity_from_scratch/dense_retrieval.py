import math


def dot_product(a, b):
    score = 0
    for i in range(len(a)):
        score += a[i] * b[i]
    return score


def vector_norm(a):
    sqr_sum = 0
    for x in a:
        sqr_sum += x ** 2
    return math.sqrt(sqr_sum)


def cosine_similarity(a, b):
    dot = dot_product(a, b)
    norma = vector_norm(a)
    normb = vector_norm(b)

    if norma == 0 or normb == 0:
        return 0.0

    return dot / (norma * normb)


def dense_search(query, documents, doc_embeddings, model, topk=3):
    query_embedding = model.encode(query)
    results = []

    for i in range(len(documents)):
        score = cosine_similarity(query_embedding, doc_embeddings[i])

        result = {
            "id": documents[i]["id"],
            "score": float(score)
        }
        results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:topk]


# 只有直接运行这个文件，才执行下面的测试
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from idf_practice import bm25

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    documents = [
        {"id": "D1", "text": "How to bake a chocolate cake"},
        {"id": "D2", "text": "Automobile maintenance and repair guide"},
        {"id": "D3", "text": "Python programming tutorial"},
        {"id": "D4", "text": "How to change engine oil in a vehicle"}
    ]

    doc_text = [doc["text"] for doc in documents]
    doc_embeddings = model.encode(doc_text)

    query = "car"

    bm25_results = bm25(
        query, documents, documents, topk=3
    )

    dense_results = dense_search(
        query, documents, doc_embeddings, model, topk=3
    )

    print("BM25:", bm25_results)
    print("Dense:", dense_results)
