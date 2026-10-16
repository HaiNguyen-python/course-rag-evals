from pathlib import Path

import pytest

from course_rag import evals
from course_rag.cli import main

QUESTIONS = Path(__file__).resolve().parents[1] / "evals" / "questions.yaml"


def test_sweep_picks_the_middle_of_the_best_range():
    # Answerable scores >= 0.5, unanswerable <= 0.3: any threshold in (0.3, 0.5] is perfect.
    result = evals.sweep_threshold(
        [0.5, 0.6, 0.9, 0.1, 0.3], [True, True, True, False, False], step=0.01, upper=1.0
    )
    assert result.best == pytest.approx(0.405, abs=0.006)
    i = result.thresholds.index(0.4)
    assert result.answer_rate[i] == 1.0 and result.refusal_rate[i] == 1.0


def test_question_kinds_are_derived_from_ids():
    kinds = {q.kind for q in evals.load_questions(QUESTIONS)}
    assert kinds == {"direct", "paraphrase", "unanswerable"}


def test_retrieval_by_kind(assistant):
    questions = evals.load_questions(QUESTIONS)
    _, answers = evals.run(assistant, questions)
    by_kind = evals.retrieval_by_kind(questions, answers)
    assert set(by_kind) == {"direct", "paraphrase"}
    assert by_kind["direct"]["n"] == 24 and by_kind["paraphrase"]["n"] == 12
    # Paraphrases are harder than questions that reuse the notes' own words.
    assert by_kind["paraphrase"]["mrr"] < by_kind["direct"]["mrr"]


def test_sweep_command_writes_chart(corpus, tmp_path, monkeypatch, capsys):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("CORPUS_DIR", str(corpus))
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    main(["index"])
    main(["sweep", "--questions", str(QUESTIONS), "--charts", str(tmp_path / "img")])
    assert "best threshold" in capsys.readouterr().out
    assert (tmp_path / "img" / "threshold_sweep.png").stat().st_size > 10_000
