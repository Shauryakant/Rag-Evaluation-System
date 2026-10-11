# Production RAG Serving: Stage Latencies and Cold Starts

Deploying real-time RAG applications on containerized infrastructure (such as Hugging Face Spaces or Render) introduces latency constraints across each execution stage.

## Stage Latency Breakdown
A production RAG request exhibits distinct per-stage latency profiles:
1. Embedding Generation: Encoding the query string with BGE-small on CPU takes 10 to 30 milliseconds.
2. First-Stage Retrieval: Querying FAISS IndexFlatIP (exact cosine) across small to medium corpora takes under 5 milliseconds. BM25 scoring takes under 8 milliseconds.
3. Second-Stage Reranking (Cross-Encoder): Scoring top-20 candidate pairs on CPU takes 60 to 180 milliseconds.
4. LLM Generation (Groq LPUs): Streaming time-to-first-token on Groq llama-3.1-8b-instant takes 120 to 250 milliseconds with generation throughput exceeding 500 tokens per second.

## Pre-warming and Cold Starts
On serverless or free-tier container hosting, cold starts can exceed 60 seconds if transformer model weights are downloaded from Hugging Face Hub on boot. Baking pre-downloaded model weights and pre-computed FAISS index caches directly into the container image eliminates runtime download stalls and bounds boot latency.
