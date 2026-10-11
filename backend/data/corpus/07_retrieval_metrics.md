# Retrieval Evaluation Metrics: Precision, Recall, MRR, and Hit Rate

Evaluating retrieval engines quantitatively requires standardized metrics that distinguish rank positioning, recall depth, and precision trade-offs.

## Standard IR Metrics
- Hit Rate@K: Proportion of queries where at least one ground-truth passage appears in the top-K retrieved list. Returns 1.0 or 0.0 per query.
- Recall@K: Proportion of all relevant ground-truth passages that appear within the top-K list. For queries with a single ground-truth passage, Recall@K equals Hit Rate@K.
- Mean Reciprocal Rank (MRR): Evaluates ranking accuracy by computing the reciprocal rank of the first relevant hit: 1 / rank. If the first relevant passage is at rank 1, MRR is 1.0; at rank 2, MRR is 0.5.

## The Structural Cap on Precision@K
Precision@K measures the proportion of retrieved passages in top-K that are relevant: relevant_in_k / k. In standard question-answering benchmarks, a question typically has only 1 or 2 relevant passages in the entire corpus. Consequently, at k=5, the theoretical maximum precision is 1/5 = 0.20 (or 2/5 = 0.40). At k=10, the maximum is 0.10. Precision@K is structurally constrained by the ground-truth cardinality and must be read alongside Recall and MRR rather than in isolation.
