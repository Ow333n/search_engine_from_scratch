import re

documents = [
    {"id": "D1", "text": "BM25 is a lexical ranking algorithm for search engines."},
    {"id": "D2", "text": "Dense retrieval uses embeddings to match semantic meaning."},
    {"id": "D3", "text": "Hybrid search combines lexical and dense retrieval."},
    {"id": "D4", "text": "Search engines rank documents for a user query."},
    {"id": "D5", "text": "Embeddings represent the meaning of sentences as vectors."}
]

def tokenizer(text):
    return re.findall(r'[a-z0-9]+', text.lower())





def keyword_score(query, docs, top_k = 2):
    scores = []

    query_token = set(tokenizer(query))

    for doc in docs:
        doc_text = doc["text"]
        doc_token = tokenizer(doc_text)

        score = 0

        for token in query_token:
            count = doc_token.count(token)
            score += count
        
        if score > 0:
            result = {'id': doc['id'], 'score': score}
            scores.append(result)
    scores.sort(key = lambda x:x['score'], reverse = True)
    return scores[:top_k]



query = "search search lexical is"
#print(keyword_score(query, documents))
