from collections import defaultdict
from inverted_index import search
from dense_retrieval import dense_search
from inverted_index import build_inverted_index
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
	'sentence-transformers/all-MiniLM-L6-v2'
)

documents = [
    {"id": "D1", "text": "How to bake a chocolate cake"},
    {"id": "D2", "text": "Automobile maintenance and repair guide"},
    {"id": "D3", "text": "Python programming tutorial"},
    {"id": "D4", "text": "How to change engine oil in a vehicle"}
]

index = build_inverted_index(documents)
doc_map = {doc['id']: doc for doc in documents}
doc_text = [doc['text'] for doc in documents]
doc_embeddings = model.encode(doc_text)

def rrf_fusion(bm25_results, dense_results, k = 60, topk = 3):
	score = defaultdict(float)

	for results in [bm25_results, dense_results]:
		for rank, item in enumerate(results, start = 1):
			doc_id = item['id']
			score[doc_id] +=  1 / (k + rank)
	
	result = []
	for doc_id, score in score.items():
		result.append({'id': doc_id, 'score': score})
	result.sort(key = lambda x:x['score'], reverse = True)
	return result[:topk]


def hybrid_search(query, index, doc_map, documents, doc_embeddings, model, topk = 3):

	bm25_results = search(query, index, doc_map, documents, topk = topk)

	dense_results = dense_search(query, documents, doc_embeddings, model, topk = topk)

	result = rrf_fusion(bm25_results, dense_results, k = 60, topk = 3)

	return result

query = "car repair"

print("BM25:")
print(search(query, index, doc_map, documents, topk=4))

print("Dense:")
print(dense_search(query, documents, doc_embeddings, model, topk=4))

print("Hybrid:")
print(hybrid_search(
    query, index, doc_map, documents,
    doc_embeddings, model, topk=3
))