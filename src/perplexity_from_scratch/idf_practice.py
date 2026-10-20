import math
import re
from collections import defaultdict
documents = [
    {"id": "D1", "text": "search engine uses data"},
    {"id": "D2", "text": "search engine ranks documents"},
    {"id": "D3", "text": "search engine uses bm25"},
    {"id": "D4", "text": "search engine uses embeddings"}
]

def tokenizer(text):
    return re.findall(r'[a-z0-9]+', text.lower())

def document_frequency(term, docs):
    df = 0
    for doc in docs:
        doc_token = set(tokenizer(doc['text']))
        if term in doc_token:
            df += 1
    return df

def idf(term, docs):
    N = len(docs)
    df = document_frequency(term, docs)
    if N == 0:
        return 0.0
    score = math.log(1 + (N - df + 0.5) / (df + 0.5))
    return score

def tf_saturated(tf, avgdl, doc_len, k = 1.2, b = 0.75):
    ld = 1 - b + b * (doc_len / avgdl)
    return tf * (k+1) / (tf + k * ld)

def cal_avgdl(docs):
    dlsum = 0
    for d in docs:
        dlen = len(tokenizer(d['text']))
        dlsum += dlen
    N = len(docs)
    if N == 0:
        return 0.0
    avgdl = dlsum / N
    return avgdl

def bm25(query, candidate_docs, corpus_docs, topk = 3):

    idf_score = defaultdict(float)
    res = []
    query_token = set(tokenizer(query))
    avgdl = cal_avgdl(corpus_docs)
    if avgdl == 0:
        return []
    for token in query_token:
        idf_score[token] = idf(token, corpus_docs)
    for doc in candidate_docs:
        score = 0
        doc_token = tokenizer(doc['text'])
        doc_len = len(doc_token)
        for token in query_token:
            tf = doc_token.count(token)
            tf_sat = tf_saturated(tf, avgdl, doc_len)
            tfidf_result = tf_sat * idf_score[token]
            score += tfidf_result
        if score > 0:
            result = {'id': doc['id'], 'score': score}
            res.append(result)
    res.sort(key = lambda x:x['score'], reverse = True)
    return res[:topk]



