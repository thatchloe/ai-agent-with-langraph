"""Knowledge agent: answers how-to / policy questions with RAG over the knowledge base.

1. Retrieve the top-k articles for the customer's message + classifier summary
   (``udahub.search_knowledge`` MCP tool).
2. If the best match is below the relevance threshold of the retrieval mode
   (lexical or hybrid, see ``tools/retrieval.py``) -> escalate (no guessing).
3. Otherwise the LLM answers grounded *only* in the retrieved articles and
   reports whether they actually answer the question; if not, or confidence is
   low, the ticket is escalated.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel, Field

from agentic.agents.common import (
    LLMProvider,
    channel_guidance,
    conversation,
    addressee,
    known_first_name,
    format_customer,
    format_memories,
    latest_user_message,
)
from agentic.logging_utils import log_event
from agentic.tools.mcp_client import call_tool
from agentic.tools.retrieval import LEXICAL_THRESHOLD

MIN_RELEVANCE = LEXICAL_THRESHOLD  # default; search_knowledge reports the threshold for its mode
MIN_ANSWER_CONFIDENCE = 0.5
TOP_K = 3


class KnowledgeAnswer(BaseModel):
    """Grounded answer for the customer."""

    answer: str = Field(description="Reply to the customer, based only on the articles.")
    answerable: bool = Field(description="True only if the articles contain the information needed.")
    confidence: float = Field(ge=0, le=1)
    used_article_titles: list[str] = Field(default_factory=list)


SYSTEM_PROMPT = """You are the Knowledge agent of UDA-Hub, answering CultPass customers.
Answer ONLY with information from the knowledge-base articles below. Do not invent policies, prices or steps.
If the articles do not cover the question, set answerable=false.
Be warm and concise (max ~120 words), address {addressee}, and use the
"suggested phrasing" from articles when it fits. Never promise refunds or exceptions.
{channel_style}

{customer}

{memory}

Knowledge-base articles:
{articles}"""


def relevant_articles(state: dict, k: int = 2) -> list[dict]:
    """Articles above the relevance threshold for the latest message (shared with the account agent)."""
    query = f"{latest_user_message(state)}\n{(state.get('classification') or {}).get('summary', '')}".strip()
    result = call_tool("udahub", "search_knowledge", {"query": query, "account_id": state["account_id"], "k": k})
    if not result.get("ok"):
        return []
    threshold = result.get("min_relevance", MIN_RELEVANCE)
    hits = result.get("results", [])
    top = hits[0]["score"] if hits else 0.0
    return [h for h in hits if h["score"] >= max(threshold, top * 0.5)]


class KnowledgeAgent:
    name = "knowledge_agent"

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def __call__(self, state: dict) -> dict:
        ticket_id = state.get("ticket_id")
        cls = state.get("classification") or {}
        user_msg = latest_user_message(state)

        if cls.get("category") == "closing":
            reply = (
                f"You're welcome{', ' + known_first_name(state) if known_first_name(state) else ''}! I'm glad I could help. "
                "I'll mark this ticket as resolved - just write again if anything else comes up."
            )
            record = log_event("answer", ticket_id, self.name, mode="closing")
            return {
                "messages": [AIMessage(reply, name=self.name)],
                "resolution": {"status": "resolved", "agent": self.name, "confidence": 1.0},
                "trace": [record],
            }

        query = f"{user_msg}\n{cls.get('summary', '')}".strip()
        result = call_tool(
            "udahub", "search_knowledge", {"query": query, "account_id": state["account_id"], "k": TOP_K}
        )
        articles = result.get("results", []) if result.get("ok") else []
        threshold = result.get("min_relevance", MIN_RELEVANCE)
        top = articles[0]["score"] if articles else 0.0
        relevant = [a for a in articles if a["score"] >= max(threshold, top * 0.5)]
        retrieval = log_event(
            "retrieval",
            ticket_id,
            self.name,
            query=query,
            hits=[{"title": a["title"], "score": a["score"]} for a in articles],
            top_score=top,
            mode=result.get("mode"),
            threshold=threshold,
        )
        if not relevant:
            reason = f"No relevant knowledge-base article (best score {top:.2f} < {threshold})"
            return {"retrieved_articles": articles, "escalation_reason": reason, "trace": [retrieval]}

        articles_text = "\n\n".join(f"### {a['title']}\n{a['content']}" for a in relevant)
        try:
            prompt = SystemMessage(
                SYSTEM_PROMPT.format(
                    addressee=addressee(state),
                    customer=format_customer(state),
                    memory=format_memories(state),
                    articles=articles_text,
                    channel_style=channel_guidance(state),
                )
            )
            structured = self.llm().with_structured_output(KnowledgeAnswer, method="function_calling")
            answer = structured.invoke([prompt, *conversation(state)])
            if not isinstance(answer, KnowledgeAnswer):
                answer = KnowledgeAnswer.model_validate(answer)
        except Exception as exc:  # noqa: BLE001
            reason = f"Knowledge agent could not generate an answer ({type(exc).__name__})"
            return {"retrieved_articles": relevant, "escalation_reason": reason, "trace": [retrieval]}

        decision = log_event(
            "answer",
            ticket_id,
            self.name,
            answerable=answer.answerable,
            confidence=answer.confidence,
            used_articles=answer.used_article_titles,
        )
        if not answer.answerable or answer.confidence < MIN_ANSWER_CONFIDENCE:
            reason = f"Knowledge base does not fully answer the question (confidence {answer.confidence:.2f})"
            return {"retrieved_articles": relevant, "escalation_reason": reason, "trace": [retrieval, decision]}

        return {
            "messages": [AIMessage(answer.answer, name=self.name)],
            "retrieved_articles": relevant,
            "resolution": {
                "status": "resolved",
                "agent": self.name,
                "confidence": answer.confidence,
                "articles": answer.used_article_titles,
            },
            "trace": [retrieval, decision],
        }


def route_after_knowledge(state: dict) -> str:
    return "escalation_agent" if state.get("escalation_reason") else "finalize"
