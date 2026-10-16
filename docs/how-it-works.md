# How it works

This page follows one question through the system and explains the important code on the way.
File references point to `src/course_rag/`.

## 1. Indexing the course material

`course-rag index` → `ingest.py → index_corpus`:

1. **Load** every `.md`, `.txt` and `.pdf` in `corpus/` (`documents.py → load_documents`).
2. **Hash** each document's text (SHA-256). If the hash matches what the store already has, the
   document is skipped: no new embeddings, no cost. Changed documents are re-embedded; documents
   that disappeared are deleted (in Postgres, chunks go with them via `ON DELETE CASCADE`).
3. **Chunk** each document by markdown section (`chunk_document`): 120 words with a 30-word
   overlap, and a chunk never crosses a heading. A citation then points to one topic, and the
   overlap keeps sentences that sit on a chunk border findable.
4. **Embed** each chunk together with its title and section heading, and store it.

Embedders (`embeddings.py`) share one interface:

| Embedder | When |
| --- | --- |
| `hashing` (default) | offline and in CI. Word and character n-grams hashed into 1,024 numbers. Deterministic, no download, but it has no idea that "untouched" means "raw". |
| `sentence-transformers` | a local neural model; understands meaning |
| `openai` | any OpenAI-compatible `/embeddings` API |

Stores (`store.py`): `MemoryStore` (numpy + JSON files in `index/`) or `PgVectorStore` (Postgres
with the pgvector extension and an HNSW index for fast cosine search).

## 2. Retrieving for a question

`retrieval.py → Retriever.retrieve("How does a type 2 slowly changing dimension keep history?")`:

1. **Vector search:** embed the question, take the 20 most similar chunks by cosine similarity.
2. **Keyword search:** BM25 (`bm25.py`, written out in ~50 lines). Rare words such as "type 2" or
   "dimension" weigh more than common ones; long chunks are normalised so they don't win by size.
3. **Fuse with Reciprocal Rank Fusion:** `score = Σ 1 / (60 + rank)` over both lists. It only
   uses ranks, so the two very different score scales never need tuning weights.
4. **Relevance score** for the refusal decision: half "share of the question's content words found
   in the best chunk", half "best cosine similarity". It is between 0 and 1.

## 3. Answer or refuse

`answer.py → Assistant.ask`:

- If relevance is below the threshold (`MIN_RELEVANCE`, default 0.238), the assistant replies
  "I can't find this in the course material." and never calls a model. That saves cost and avoids
  confident wrong answers.
- Otherwise a generator writes the answer from the top 4 chunks:
  - `ExtractiveGenerator` (default): picks the 2 sentences that share the most words with the
    question, and cites their chunks.
  - `OpenAICompatibleGenerator`: sends a prompt that says *answer only from the context, cite chunk
    ids in square brackets, otherwise refuse with this exact sentence*. Works with OpenAI, Ollama,
    vLLM and similar.
- **Citations are checked:** every `[chunk-id]` in the answer must be one of the retrieved chunks.
  Anything else is reported as an invalid (hallucinated) citation.
- Token counts (from the API, or estimated) become a cost with the prices in `.env`.

## 4. Choosing the threshold from data

`course-rag sweep` → `evals.py → sweep_threshold`:

For every threshold from 0 to 0.6, it computes how many course questions would still be answered
and how many off-topic questions would be refused. The best threshold maximises balanced accuracy
(both rates count equally, even though there are 36 course questions and only 6 off-topic ones).
Several thresholds tie, so it takes the middle of the best range: the largest safety margin on
both sides. Result: 0.238. The old hand-picked 0.20 answered "Which Python web framework is
fastest?" from the Airflow notes, because "Python" matched; 0.238 refuses it and still answers 94%
of course questions.

With only 6 off-topic questions this is a small sample, so the threshold should be re-checked on
more questions before relying on it.

## 5. Evaluating every change

`course-rag eval --compare` → `evals.py`:

- **42 questions** (`evals/questions.yaml`): 24 *direct* (reuse the notes' words), 12
  *paraphrase* (same meaning, different words), 6 *unanswerable*.
- **Retrieval metrics:** hit rate@4 (is a right document in the top 4?) and MRR (how high is the
  first right chunk?), overall and per question type.
- **Answer metrics:** answer rate, correct refusal rate, citation validity, grounding (cites a right
  document), faithfulness (share of answer words found in the cited chunks).
- **Speed and cost:** p50/p95 latency, cost per 1,000 questions.
- **Gate:** `evals/thresholds.yaml` sets minimums; CI fails the build if the hybrid setup drops
  below any of them.

## 6. Serving and tracing

- `api.py`: `POST /ask {"question": "..."}` → answer, citations, sources with scores, latency, cost.
  The assistant and the tracer are FastAPI dependencies, so tests replace them easily.
- `tracing.py`: every question is logged to SQLite with latency, tokens, cost, what was retrieved and
  what was cited. `course-rag traces` prints a summary.
