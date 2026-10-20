from collections import defaultdict
from idf_practice import tokenizer
from idf_practice import bm25
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

index = build_inverted_index(documents)


def retrieve_candidates(query, index):
    query_token = set(tokenizer(query))
    candidate_ids = set()
    for token in query_token:
        posting = index.get(token, {})
        for doc_id in posting:
            candidate_ids.add(doc_id)
    return sorted(candidate_ids)

doc_map = {doc['id']: doc for doc in documents}
def search(query, index, doc_map, corpus_docs, topk = 3):
    candidate_ids = retrieve_candidates(query, index)

    

    candidate_docs = [doc_map[ids] for ids in candidate_ids]

    results = bm25(query, candidate_docs, documents, topk = topk)

    return results

print(search("bm25", index, doc_map, documents, topk=3))
print(search("bm25 search", index, doc_map, documents, topk=2))
print(search("embeddings", index, doc_map, documents, topk=3))
print(search("python", index, doc_map, documents, topk=3))