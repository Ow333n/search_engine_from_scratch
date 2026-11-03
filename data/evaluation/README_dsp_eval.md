# DSP Search Evaluation — DRAFT

This is a **proposed**, not yet adjudicated, query/qrel seed set for the KCL Course Search project.

- Corpus: 50 PDFs, 447 nonempty extracted pages, 468 chunks; chunk size 100 words, overlap 20.
- Queries: 58, across 10 subject groups.
- Proposed relevance judgments: 123 (only grade 2/3 candidates; other documents are **unjudged**, not grade 0).
- IDs are the actual exported `D..._C...` IDs; adding/reordering PDFs or changing chunking invalidates them.
- Lecture versions such as `FDSP22.pdf` and `FDSP22S.pdf` overlap substantially. Keep them fixed until the first evaluation pass; mark all relevant variants or deduplicate and regenerate IDs **before** finalizing labels.
- Technical parsing note: mathematical symbols in extracted text are sometimes incomplete. Verify source pages while reviewing.
- Do not treat preliminary Recall/MRR/NDCG as ground-truth performance. Use pooling (BM25 + Dense + Hybrid + reranker candidates) to discover additional relevant chunks and review them.

## Files

- `queries.json`: questions and dev/test split.
- `qrels_proposed.json`: provisional 0–3 judgments keyed by query ID and actual chunk ID.
- `annotation_review.csv`: exactly what needs source-page verification.
- `dataset_report.json`: counts, coverage, and evaluation cautions.
- `validate_dsp_eval.py`: structural check against your original `dsp_chunks.json`.

## Local use

Save the files into `data/evaluation/` alongside the exported `dsp_chunks.json`, then run:

```bash
uv run python data/evaluation/validate_dsp_eval.py
```

After human validation, save a final `qrels.json`. Before measuring NDCG, compute IDCG from *all* reviewed relevant chunks for a query, not from just the retrieved ranked list. For Recall/MRR use relevance >= 2, with consistent and documented treatment of unjudged results.

**Review priority:** inspect all proposed grades for a few queries in each topic group, especially formula-heavy queries, and build a judgement pool from all four retrievers before claiming complete coverage.
