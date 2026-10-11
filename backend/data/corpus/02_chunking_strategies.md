# Chunking Strategies: Size, Overlap, and Boundary Awareness

Chunking transforms continuous documents into atomic passages suitable for embedding models with constrained token context lengths. Choosing chunking parameters directly impacts downstream retrieval quality and token utilization.

## Chunk Size Trade-Offs
Small chunks (100 to 250 characters) capture fine-grained, localized facts with minimal extraneous noise. However, they frequently fragment complex explanations and drop surrounding context. Large chunks (800 to 1500 characters) provide complete semantic narratives but dilute sentence-level embedding vectors and introduce irrelevant tokens into the LLM context window. A default chunk size of 500 characters balances conceptual completeness and embedding specificity.

## Overlap and Boundary Preservation
Fixed-window chunking introduces arbitrary splits that can sever sentences or key entity definitions. Implementing a sliding window overlap (typically 10% of chunk size, such as 50 characters) ensures that boundary-spanning sentences appear completely in at least one chunk. Heading-aware recursive splitting further improves coherence by preferring splits along Markdown headers (##, ###) and paragraph boundaries before resorting to character-level slicing.
