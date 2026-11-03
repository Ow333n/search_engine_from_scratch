import math

def recall_at_k(retrieved_ids, relevant_ids, k):
	if not retrieved_ids or not relevant_ids:
		return 0.0

	top_k_docs = retrieved_ids[:k]

	intersection_count = len(set(relevant_ids) & set(top_k_docs))

	return intersection_count / len(relevant_ids)

def reciprocal_rank_at_k(retrieved_ids, relevant_ids, k):
	rank_score = 0
	for i, x in enumerate(retrieved_ids[:k], start = 1):
		if x in relevant_ids:
			rank_score += 1 / i
			break
	
	return rank_score


def mean_reciprocal_rank(queries, k):
	scores = 0
	for query in queries:
		scores += reciprocal_rank_at_k(query['retrieved'], query['relevant'], k)
	
	if not queries:
		return 0.0
	scores = scores / len(queries)

	return scores


def dcg_at_k(relevances, k):
	score = 0
	for i, x in enumerate(relevances[:k], start=1):
		score += (2 ** x - 1) / (math.log2(i + 1))
	return score


def ndcg_at_k(relevances, k):
	ideal_score = 0
	actual_score = dcg_at_k(relevances, k)

	ideal_relevance = sorted(relevances, reverse = True)
	ideal_score = dcg_at_k(ideal_relevance, k)

	if ideal_score == 0:
		return 0.0
	
	return actual_score / ideal_score