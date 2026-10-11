# Sparse Lexical Retrieval: The BM25 Algorithm

BM25 (Best Matching 25) is a probabilistic ranking function that evaluates the relevance of documents to a given query based on term frequency and document length normalization.

## Mathematical Formulation
BM25 builds on classical TF-IDF principles through two tuning parameters: k1 (term saturation) and b (document length penalty).
- Term Frequency Saturation (k1, typically 1.5): Governs how quickly the score saturates as a query term repeats within a passage. Unlike linear term frequency, BM25 prevents single-term dominance.
- Length Normalization (b, typically 0.75): Penalizes overly verbose passages by scaling term frequency relative to average document length across the corpus.

## Advantages Over Dense Retrieval
Dense vector models often fail on rare tokens, part numbers, domain acronyms, and proper nouns that appear infrequently in pre-training data. BM25 provides exact keyword retrieval precision, making it an essential complement to semantic vector search.
