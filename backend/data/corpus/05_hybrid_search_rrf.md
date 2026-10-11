# Hybrid Retrieval and Reciprocal Rank Fusion

Hybrid search merges dense semantic retrieval and sparse lexical BM25 retrieval to leverage the complementary strengths of both modalities.

## The Calibration Problem
Dense cosine similarities range from 0.0 to 1.0 (or -1.0 to 1.0), whereas BM25 produces unbounded positive floating-point scores dependent on corpus frequency statistics. Linear score combination (alpha * dense + (1 - alpha) * sparse) requires extensive empirical parameter tuning and is brittle when queries vary in token specificity.

## Reciprocal Rank Fusion (RRF)
Reciprocal Rank Fusion bypasses score scale discrepancies by aggregating rank positions rather than raw scores. For each candidate document d across rank lists M:
RRF_Score(d) = sum(1 / (k + rank_m(d))) for m in M

The smoothing constant k (conventionally set to 60) dampens the impact of top-ranked outliers, ensuring that documents appearing consistently high across both dense and sparse lists receive the highest composite score.
