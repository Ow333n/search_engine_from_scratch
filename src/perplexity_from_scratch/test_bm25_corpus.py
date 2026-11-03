
from inverted_index import build_inverted_index, search
from idf_practice import bm25

docs = [
    {"id": "A", "text": "search search retrieval"},
    {"id": "B", "text": "dense retrieval ranking"},
    {"id": "C", "text": "course notes and lecture materials"},
]

query = "search"

index = build_inverted_index(docs)
doc_map = {doc["id"]: doc for doc in docs}

# 使用正确的完整 Corpus 作为参考
expected = bm25(query, docs, docs, topk=3)

# 检查现有搜索接口
actual = search(query, index, doc_map, docs, topk=3)

assert [x["id"] for x in actual] == [
    x["id"] for x in expected
]

assert abs(
    actual[0]["score"] - expected[0]["score"]
) < 1e-9

print("PASS: BM25 corpus consistency")
