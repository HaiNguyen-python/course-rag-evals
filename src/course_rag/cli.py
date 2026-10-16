"""CLI: course-rag index | ask | eval | serve | traces"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from .config import get_settings

DEFAULT_THRESHOLDS = {
    "hit_rate_at_k": 0.85,
    "answer_rate": 0.80,
    "refusal_rate": 0.80,
    "citation_ok": 0.95,
    "grounded": 0.80,
}


def cmd_index(args) -> None:
    from .embeddings import make_embedder
    from .factory import make_store
    from .ingest import index_corpus

    s = get_settings()
    embedder = make_embedder(s.embedder, base_url=s.llm_base_url, api_key=s.llm_api_key)
    stats = index_corpus(
        s.corpus_dir, make_store(s, embedder.dim), embedder, s.chunk_words, s.chunk_overlap
    )
    print(json.dumps(asdict(stats), indent=2))


def cmd_ask(args) -> None:
    from .factory import make_assistant
    from .tracing import Tracer

    s = get_settings()
    assistant = make_assistant(s, mode=args.mode)
    a = assistant.ask(args.question)
    Tracer(s.traces_path).log(a, assistant.generator.name)
    print(a.text)
    print(
        f"\nsources: {', '.join(c for c, _ in a.retrieved)}  "
        f"relevance={a.relevance}  {a.latency_ms} ms  ${a.cost_usd:.6f}"
    )


def cmd_eval(args) -> None:
    from . import evals
    from .factory import make_assistant

    s = get_settings()
    questions = evals.load_questions(Path(args.questions))
    modes = ["vector", "bm25", "hybrid"] if args.compare else [args.mode]
    results, answers, by_kind = {}, {}, {}
    for mode in modes:
        results[mode], answers[mode] = evals.run(make_assistant(s, mode=mode), questions)
        by_kind[mode] = evals.retrieval_by_kind(questions, answers[mode])
    evals.write_report(results, answers, Path(args.out))
    print(evals.to_markdown(results))
    print("\nRetrieval by question type (MRR):")
    for mode, kinds in by_kind.items():
        print(f"  {mode:7s} " + "  ".join(f"{k}={v['mrr']:.2f}" for k, v in kinds.items()))
    if args.charts:
        from .charts import plot_mrr_by_kind

        plot_mrr_by_kind(by_kind, Path(args.charts) / "mrr_by_question_type.png")

    thresholds = DEFAULT_THRESHOLDS
    if args.thresholds and Path(args.thresholds).exists():
        import yaml

        thresholds = yaml.safe_load(Path(args.thresholds).read_text())
    gate_mode = "hybrid" if "hybrid" in results else modes[0]
    failures = evals.check_thresholds(results[gate_mode], thresholds)
    if failures:
        print("\nEVAL GATE FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print(f"\nEval gate passed for '{gate_mode}'.")


def cmd_sweep(args) -> None:
    """Pick the refusal threshold from the eval set (see evals.sweep_threshold)."""
    from . import evals
    from .factory import make_assistant

    s = get_settings()
    questions = evals.load_questions(Path(args.questions))
    retriever = make_assistant(s, mode="hybrid").retriever
    relevances = [retriever.retrieve(q.question, s.top_k).relevance for q in questions]
    result = evals.sweep_threshold(relevances, [q.answerable for q in questions])
    i = result.thresholds.index(min(result.thresholds, key=lambda t: abs(t - result.best)))
    print(
        f"best threshold {result.best:.3f}: answers {result.answer_rate[i]:.0%} of answerable, "
        f"refuses {result.refusal_rate[i]:.0%} of unanswerable (current {s.min_relevance})"
    )
    if args.charts:
        from .charts import plot_sweep

        plot_sweep(result, args.previous, Path(args.charts) / "threshold_sweep.png")


def cmd_traces(args) -> None:
    from .tracing import Tracer

    print(json.dumps(Tracer(get_settings().traces_path).summary(), indent=2))


def cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run("course_rag.api:app", host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="course-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("index", help="(re)index the corpus incrementally").set_defaults(func=cmd_index)

    p = sub.add_parser("ask", help="ask one question")
    p.add_argument("question")
    p.add_argument("--mode", default="hybrid", choices=["vector", "bm25", "hybrid"])
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="run the evaluation set")
    p.add_argument("--questions", default="evals/questions.yaml")
    p.add_argument("--thresholds", default="evals/thresholds.yaml")
    p.add_argument("--mode", default="hybrid", choices=["vector", "bm25", "hybrid"])
    p.add_argument("--compare", action="store_true", help="compare vector, bm25 and hybrid")
    p.add_argument("--out", default="reports")
    p.add_argument("--charts", help="folder for PNG charts (needs matplotlib)")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("sweep", help="choose the refusal threshold from the eval set")
    p.add_argument("--questions", default="evals/questions.yaml")
    p.add_argument("--charts", help="folder for the PNG chart (needs matplotlib)")
    p.add_argument("--previous", type=float, default=0.20, help="old threshold to mark")
    p.set_defaults(func=cmd_sweep)

    sub.add_parser("traces", help="summary of logged questions").set_defaults(func=cmd_traces)

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
