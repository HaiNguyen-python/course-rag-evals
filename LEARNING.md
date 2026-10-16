# Learning guide (for me, Hai)

Goal: explain how a RAG system works under the hood, why evaluation matters more than the model,
and defend every number in the README.

Start with [docs/how-it-works.md](docs/how-it-works.md) (the code walkthrough) and
[docs/decisions.md](docs/decisions.md) (the reasons). This file is the practice plan.

## 1. Run it (1 hour)

```bash
make install && make test
make compare
course-rag ask "What is a data contract?"
course-rag ask "What is the capital of Australia?"     # should refuse
course-rag traces
course-rag sweep --charts docs/img                      # how the refusal threshold was chosen
```

Open `reports/eval_results.json` and find the questions each setup got wrong. Explain each miss.

## 2. Read the code in this order

| File | Understand | Ask yourself |
| --- | --- | --- |
| `documents.py` | Section-aware chunking with overlap | What breaks if chunks are too big? Too small? |
| `bm25.py` | idf, term frequency, length normalisation (k1, b) | Why does a rare word score higher? |
| `embeddings.py` | Hashing trick vs neural embeddings | Why can't hashing match "untouched" with "raw"? |
| `retrieval.py` | RRF fusion, relevance score | Why fuse ranks instead of raw scores? |
| `ingest.py` | Content hashes for incremental indexing | What happens when a file is deleted? |
| `answer.py` | Refusal threshold, citations, prompt, cost | How do you detect a hallucinated citation? |
| `evals.py` | Hit rate, MRR, grounding, faithfulness proxy, per-type metrics, threshold sweep | Which metric would catch a bad prompt change? Why balanced accuracy for the sweep? |
| `store.py` (PgVectorStore) | pgvector, HNSW index, cosine distance `<=>` | Why HNSW instead of an exact scan at scale? |

## 3. Concepts in my own words (fill in)

- **RAG:** ...
- **Chunking and overlap:** ...
- **BM25:** ...
- **Embeddings and cosine similarity:** ...
- **Reciprocal Rank Fusion:** ...
- **Hit rate@k and MRR:** ...
- **Grounding vs faithfulness:** ...
- **Balanced accuracy and why the threshold is taken from the middle of the best range:** ...
- **Why tuning on the reported set is a risk (overfitting the eval):** ...

## 4. Interview questions this repo answers

1. *How would you evaluate an LLM feature?* A fixed question set with expected sources, retrieval
   metrics separately from answer metrics, refusal tests, a CI gate, and traces in production.
2. *How do you keep the index fresh?* Content hashes per document; re-embed only what changed;
   cascade deletes.
3. *How do you control cost?* Token counts per request, price config, cost per 1,000 questions in
   the eval report; refuse early (no LLM call) when retrieval is weak.
4. *Why hybrid search?* Keywords catch exact terms, vectors catch meaning; RRF combines them simply.
5. *How do you stop hallucinations?* Answer only from context, cite chunk ids, validate citations,
   refuse below a relevance threshold, measure faithfulness.

## 5. Make it mine (do at least three)

- [ ] `pip install ".[neural]"`, set `EMBEDDER=sentence-transformers`, rerun `make compare`, and add
      the row to the README. Do the paraphrase misses go away?
- [ ] Run `make up-llm` with Ollama and compare extractive vs LLM answers on faithfulness.
- [ ] Add 10 Vietnamese questions and a multilingual model.
- [ ] Write 15 more off-topic questions in a separate file, re-run the sweep on it, and report
      whether 0.238 still holds.
- [ ] Replace the sample notes with my own lesson material.

## 6. Honesty checklist before pushing

- [ ] I can explain BM25 and RRF on a whiteboard.
- [ ] README numbers are from my own runs.
- [ ] I can say clearly why faithfulness is 1.0 for the extractive generator.

Note: the Docker setup (`make up`, `make up-llm`) was not run where this repo was first built.
The Python code, including the pgvector store against a real Postgres, is tested.
