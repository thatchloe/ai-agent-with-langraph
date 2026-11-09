"""
Offline tests (no API key needed) for schemas, the calculator tool, prompt
selection and workflow wiring.

Run with:  python tests/test_offline.py     (or: pytest tests/)
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pydantic import ValidationError

from schemas import AnswerResponse, UserIntent
from tools import ToolLogger, create_calculator_tool
from prompts import (
    get_chat_prompt_template, get_intent_classification_prompt,
    QA_SYSTEM_PROMPT, SUMMARIZATION_SYSTEM_PROMPT, CALCULATION_SYSTEM_PROMPT,
)
from agent import create_workflow


def _raises_validation_error(fn) -> bool:
    try:
        fn()
    except ValidationError:
        return True
    return False


# --- Schemas -----------------------------------------------------------------

def test_answer_response_valid_and_defaults():
    r = AnswerResponse(question="Q?", answer="A.", confidence=0.8)
    assert r.sources == []           # default is an empty list
    assert r.timestamp is not None   # default timestamp is set automatically


def test_answer_response_confidence_bounds():
    assert _raises_validation_error(
        lambda: AnswerResponse(question="Q", answer="A", confidence=1.5))
    assert _raises_validation_error(
        lambda: AnswerResponse(question="Q", answer="A", confidence=-0.1))


def test_answer_response_type_enforcement():
    assert _raises_validation_error(
        lambda: AnswerResponse(question="Q", answer="A", confidence="high"))
    assert _raises_validation_error(
        lambda: AnswerResponse(question="Q", answer="A", confidence=0.5, sources="INV-001"))


def test_user_intent_valid_types():
    for intent in ("qa", "summarization", "calculation", "unknown"):
        assert UserIntent(intent_type=intent, confidence=0.9, reasoning="r").intent_type == intent


def test_user_intent_rejects_invalid_type_and_confidence():
    assert _raises_validation_error(
        lambda: UserIntent(intent_type="weather", confidence=0.9, reasoning="r"))
    assert _raises_validation_error(
        lambda: UserIntent(intent_type="qa", confidence=2, reasoning="r"))


# --- Calculator tool -----------------------------------------------------------

def _calculator():
    return create_calculator_tool(ToolLogger(logs_dir=tempfile.mkdtemp()))


def test_calculator_returns_string_numbers():
    calc = _calculator()
    assert calc.invoke({"expression": "2 + 3"}) == "5"
    assert calc.invoke({"expression": "69300 + 214500"}) == "283800"
    assert calc.invoke({"expression": "10 / 4"}) == "2.5"
    assert calc.invoke({"expression": "$1,200 * 0.1"}) == "120"
    assert calc.invoke({"expression": "2 ** 10"}) == "1024"


def test_calculator_rejects_unsafe_input():
    calc = _calculator()
    for expr in ["__import__('os').system('ls')", "open('x')", "abc", "9**9**9**9", ""]:
        assert calc.invoke({"expression": expr}).startswith("Error"), expr


def test_calculator_handles_math_errors():
    calc = _calculator()
    assert "division by zero" in calc.invoke({"expression": "1/0"})
    assert "invalid expression syntax" in calc.invoke({"expression": "(1+"})


def test_calculator_logs_usage():
    logger = ToolLogger(logs_dir=tempfile.mkdtemp())
    calc = create_calculator_tool(logger)
    calc.invoke({"expression": "1 + 1"})
    calc.invoke({"expression": "1 / 0"})
    logs = logger.get_logs()
    assert [log["tool_name"] for log in logs] == ["calculator", "calculator"]
    assert os.path.exists(logger.log_file)


# --- Prompts -----------------------------------------------------------------

def test_chat_prompt_selection():
    expected = {
        "qa": QA_SYSTEM_PROMPT,
        "summarization": SUMMARIZATION_SYSTEM_PROMPT,
        "calculation": CALCULATION_SYSTEM_PROMPT,
        "unknown": QA_SYSTEM_PROMPT,  # fallback
    }
    for intent, system_prompt in expected.items():
        template = get_chat_prompt_template(intent)
        assert template.messages[0].prompt.template == system_prompt
        assert set(template.input_variables) == {"input", "chat_history"}


def test_intent_prompt_formats():
    text = get_intent_classification_prompt().format(
        user_input="Summarize all contracts", conversation_history="None")
    assert "Summarize all contracts" in text


# --- Workflow ----------------------------------------------------------------

def test_workflow_structure():
    workflow = create_workflow(llm=None, tools=[])
    graph = workflow.get_graph()
    nodes = set(graph.nodes)
    assert {"classify_intent", "qa_agent", "summarization_agent",
            "calculation_agent", "update_memory"} <= nodes
    edges = {(e.source, e.target) for e in graph.edges}
    for agent in ("qa_agent", "summarization_agent", "calculation_agent"):
        assert ("classify_intent", agent) in edges
        assert (agent, "update_memory") in edges
    assert ("update_memory", "__end__") in edges
    assert workflow.checkpointer is not None


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
