# Design decisions

## 1. Evaluation before features

- **Decision:** a fixed 42-question eval set with expected documents and a CI gate.
- **Why:** without numbers every change to chunking, retrieval or prompts is a guess. The gate turns
  "it feels better" into a measured before/after for every metric.
- **Cost:** the set must be maintained, and 42 questions is small: differences of one or two
  questions are noise.

## 2. Hybrid retrieval with Reciprocal Rank Fusion

- **Decision:** vector search + BM25, fused by rank.
- **Why:** keywords win on exact terms (dbt, SCD2, PSI), vectors on meaning; RRF needs no weight
  tuning. Measured: hybrid has the best MRR on paraphrased questions (0.88 vs 0.82).

## 3. Refuse below a data-chosen threshold

- **Decision:** threshold 0.238 from `course-rag sweep`, not a guess.
- **Why:** refusal rate on off-topic questions went from 83% to 100% while course questions
  answered stayed at 94%.
- **Cost:** tuned on the same 42 questions it is reported on; more unanswerable questions are
  needed for a fair estimate (planned: a separate held-out set).

## 4. Validate citations

- **Decision:** cited chunk ids must be among the retrieved chunks.
- **Why:** an LLM can invent sources. Checking them is cheap and catches it every time.

## 5. Incremental indexing with content hashes

- **Decision:** SHA-256 per document; skip unchanged, re-embed changed, delete removed.
- **Why:** with a paid embedding API, re-embedding everything on every change wastes money and time.

## 6. Section-aware chunks with overlap

- **Decision:** 120 words, 30 overlap, never across a heading.
- **Why:** one chunk = one topic, so citations are precise; overlap keeps border sentences findable.
- **Next:** compare 80 / 120 / 200 words in the eval report.

## 7. Offline defaults

- **Decision:** hashing embeddings and an extractive generator by default.
- **Why:** CI and anyone cloning the repo can run everything without keys or model downloads.
- **Cost:** weaker on paraphrases, and faithfulness is trivially 1.0 for extractive answers; it
  becomes meaningful with an LLM generator.

## 8. pgvector, not a separate vector database

- **Decision:** Postgres + pgvector with an HNSW index.
- **Why:** one database for documents, chunks and vectors, transactions and cascading deletes for
  free, and a technology most data teams already run.
