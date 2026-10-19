from collections import defaultdict
from idf_practice import tokenizer
documents = [
    {"id": "D1", "text": "search engine uses data"},
    {"id": "D2", "text": "search engine ranks documents"},
    {"id": "D3", "text": "search engine uses bm25"},
    {"id": "D4", "text": "search engine uses embeddings"}
]
def build_inverted_index(docs):
	index = defaultdict(dict)
	for doc in docs:
		doc_id = doc['id']
		doc_token = tokenizer(doc['text'])
		for token in set(doc_token):
			count = doc_token.count(token)
			index[token][doc_id] = count
	return index


test_docs = [
    {
        "id": "D1",
        "text": "search search search engine"
    },
    {
        "id": "D2",
        "text": "search engine uses bm25"
    }
]

index = build_inverted_index(test_docs)

print(index["search"])
print(index["engine"])
print(index["bm25"])

