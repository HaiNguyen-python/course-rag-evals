import json

import httpx

from course_rag.answer import REFUSAL, Assistant, OpenAICompatibleGenerator


def fake_llm(reply: str, usage: dict | None = None):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        body = {"choices": [{"message": {"content": reply}}]}
        if usage:
            body["usage"] = usage
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler), seen


def test_llm_answer_citations_and_cost(assistant):
    transport, seen = fake_llm(
        "Bronze keeps raw data as it arrived [01-medallion-architecture#1].",
        usage={"prompt_tokens": 1000, "completion_tokens": 50},
    )
    gen = OpenAICompatibleGenerator("http://llm/v1", "key", "test-model", transport=transport)
    llm = Assistant(
        assistant.retriever, gen, min_relevance=0.2, price_in_per_m=0.15, price_out_per_m=0.60
    )
    a = llm.ask("What does the bronze layer store?")
    assert a.citations == ["01-medallion-architecture#1"]
    assert abs(a.cost_usd - (1000 * 0.15 + 50 * 0.60) / 1e6) < 1e-12
    # The prompt must contain the retrieved context with chunk ids.
    assert "[01-medallion-architecture#" in seen["body"]["messages"][1]["content"]
    assert seen["body"]["temperature"] == 0


def test_llm_refusal_and_hallucinated_citation_are_detected(assistant):
    transport, _ = fake_llm(REFUSAL)
    gen = OpenAICompatibleGenerator("http://llm/v1", "key", "m", transport=transport)
    a = Assistant(assistant.retriever, gen).ask("What does the bronze layer store?")
    assert a.refused and a.input_tokens > 0  # estimated when the API returns no usage

    transport, _ = fake_llm("Made up [99-not-real#0].")
    gen = OpenAICompatibleGenerator("http://llm/v1", "key", "m", transport=transport)
    a = Assistant(assistant.retriever, gen).ask("What does the bronze layer store?")
    assert a.extra["invalid_citations"] == ["99-not-real#0"]
