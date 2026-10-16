import json

import pytest

from course_rag.cli import main


@pytest.fixture()
def env(corpus, tmp_path, monkeypatch):
    monkeypatch.setenv("CORPUS_DIR", str(corpus))
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("TRACES_PATH", str(tmp_path / "traces.db"))
    monkeypatch.chdir(tmp_path.parent)
    return tmp_path


def test_index_ask_eval_and_traces(env, capsys, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    main(["index"])
    assert len(json.loads(capsys.readouterr().out)["added"]) == 6
    main(["index"])  # second run: nothing to do
    assert len(json.loads(capsys.readouterr().out)["unchanged"]) == 6

    main(["ask", "What is a data contract?"])
    assert "03-data-quality#" in capsys.readouterr().out

    main(
        [
            "eval",
            "--compare",
            "--questions",
            str(root / "evals" / "questions.yaml"),
            "--thresholds",
            str(root / "evals" / "thresholds.yaml"),
            "--out",
            str(env / "reports"),
            "--charts",
            str(env / "img"),
        ]
    )
    out = capsys.readouterr().out
    assert "| hybrid |" in out and "Eval gate passed" in out
    assert (env / "reports" / "eval_results.json").exists()
    assert (env / "img" / "mrr_by_question_type.png").exists()

    main(["traces"])
    assert json.loads(capsys.readouterr().out)["questions"] == 1


def test_eval_gate_fails_when_thresholds_are_too_high(env, tmp_path, capsys):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    strict = tmp_path / "strict.yaml"
    strict.write_text("mrr: 1.01\n")
    main(["index"])
    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "eval",
                "--questions",
                str(root / "evals" / "questions.yaml"),
                "--thresholds",
                str(strict),
                "--out",
                str(tmp_path / "r"),
            ]
        )
    assert exit_info.value.code == 1
    assert "EVAL GATE FAILED" in capsys.readouterr().out
