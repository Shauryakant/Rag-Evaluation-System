# Dense Embeddings: Bi-Encoder Representations and FAISS Indexing

Dense embedding models project arbitrary natural language strings into continuous, fixed-dimensional vector spaces where geometric distance reflects semantic similarity.

## Bi-Encoder Architecture
Models such as BAAI/bge-small-en-v1.5 utilize a BERT-based bi-encoder architecture. The encoder processes queries and passages independently, outputting a 384-dimensional dense representation vector through mean pooling over final token hidden states. Because passages are encoded offline into an index, query-time latency is restricted to a single forward pass over the user query followed by matrix vector multiplication.

## Normalization and Cosine Similarity
By applying L2-normalization to all passage and query embedding vectors, the Euclidean inner product mathematically reduces to cosine similarity: dot(u, v) = cos(theta). This property enables the use of FAISS IndexFlatIP (Inner Product), which calculates exact cosine similarities without the computational overhead of square-root distance normalizations during query execution.
