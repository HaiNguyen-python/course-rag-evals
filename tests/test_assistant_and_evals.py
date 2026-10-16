from pathlib import Path

from fastapi.testclient import TestClient

from course_rag import api, evals
from course_rag.answer import REFUSAL

QUESTIONS = Path(__file__).resolve().parents[1] / "evals" / "questions.yaml"


def test_answers_with_valid_citations(assistant):
    a = assistant.ask("What does the bronze layer store?")
    assert not a.refused
    assert a.citations and all(c.startswith("01-medallion-architecture#") for c in a.citations)
    assert a.extra["invalid_citations"] == []
    assert "raw data" in a.text


def test_refuses_off_topic_questions(assistant):
    a = assistant.ask("What is the capital of Australia?")
    assert a.refused and a.text == REFUSAL and a.citations == []
    # A single shared word ("Python" appears in the Airflow notes) is not enough to answer.
    assert assistant.ask("Which Python web framework is fastest?").refused


def test_eval_gate_passes_on_the_bundled_set(assistant):
    metrics, answers = evals.run(assistant, evals.load_questions(QUESTIONS))
    failures = evals.check_thresholds(
        metrics, {"hit_rate_at_k": 0.85, "refusal_rate": 0.8, "citation_ok": 0.95, "grounded": 0.8}
    )
    assert failures == [], failures
    assert len(answers) == 42


def test_check_thresholds_reports_failures():
    assert evals.check_thresholds({"mrr": 0.5}, {"mrr": 0.8}) == ["mrr: 0.500 < 0.8"]


def test_api_ask_and_health(assistant, tmp_path):
    from course_rag.tracing import Tracer

    tracer = Tracer(tmp_path / "traces.db")
    api.app.dependency_overrides[api.get_assistant] = lambda: assistant
    api.app.dependency_overrides[api.get_tracer] = lambda: tracer
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "ok"
    resp = client.post("/ask", json={"question": "What is a data contract?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["citations"] and not body["refused"]
    assert tracer.summary()["questions"] == 1
    assert client.post("/ask", json={"question": "x"}).status_code == 422
    api.app.dependency_overrides.clear()
