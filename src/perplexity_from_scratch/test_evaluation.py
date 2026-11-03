
from evaluation import recall_at_k

retrieved = ["D2", "D1", "D5", "D3", "D4"]
relevant = {"D2", "D5", "D8"}

assert abs(recall_at_k(retrieved, relevant, 5) - 2/3) < 1e-9
assert abs(recall_at_k(retrieved, relevant, 2) - 1/3) < 1e-9
assert recall_at_k(retrieved, relevant, 1) == 1/3
assert recall_at_k(retrieved, set(), 5) == 0.0

print("PASS: Recall@K")


from evaluation import reciprocal_rank_at_k

relevant = {"D2", "D5"}

assert reciprocal_rank_at_k(
    ["D2", "D1", "D5"], relevant, 3
) == 1.0

assert reciprocal_rank_at_k(
    ["D1", "D2", "D5"], relevant, 3
) == 0.5

assert abs(reciprocal_rank_at_k(
    ["D1", "D3", "D5"], relevant, 3
) - 1/3) < 1e-9

assert reciprocal_rank_at_k(
    ["D1", "D3", "D5"], relevant, 2
) == 0.0

assert reciprocal_rank_at_k(
    [], relevant, 3
) == 0.0

print("PASS: Reciprocal Rank@K")



from evaluation import mean_reciprocal_rank

queries = [
    {
        "retrieved": ["D2", "D1", "D3"],
        "relevant": {"D2"}
    },
    {
        "retrieved": ["D1", "D5", "D3"],
        "relevant": {"D5"}
    },
    {
        "retrieved": ["D1", "D3", "D4"],
        "relevant": {"D8"}
    }
]

# RR values: 1.0, 0.5, 0.0
expected = (1.0 + 0.5 + 0.0) / 3

assert abs(
    mean_reciprocal_rank(queries, 3) - expected
) < 1e-9

assert mean_reciprocal_rank([], 3) == 0.0

print("PASS: MRR@K")


from evaluation import dcg_at_k
import math

expected = 7 + 3 / math.log2(3)

assert abs(dcg_at_k([3, 2, 0], 3) - expected) < 1e-9
assert dcg_at_k([0, 0, 0], 3) == 0.0
assert dcg_at_k([], 3) == 0.0
assert dcg_at_k([3, 2, 0], 1) == 7.0

print("PASS: DCG@K")



from evaluation import ndcg_at_k

assert abs(ndcg_at_k([3, 2, 1, 0], 4) - 1.0) < 1e-9

assert ndcg_at_k([0, 0, 0], 3) == 0.0

assert abs(
    ndcg_at_k([1, 3, 0, 2], 4) - 0.7142221296584441
) < 1e-9

assert abs(ndcg_at_k([1, 3, 0, 2], 1) - 1/7) < 1e-9

print("PASS: NDCG@K")

relevances = [1, 3, 0, 2]
original = relevances.copy()

ndcg_at_k(relevances, 4)

assert relevances == original

print("PASS: NDCG input unchanged")