"""RAG retrieval quality and threshold calibration."""
import pytest
from langchain_core.embeddings import Embeddings

from agentic.agents.knowledge import MIN_RELEVANCE
from agentic.tools import udahub_tools
from agentic.tools.retrieval import Document, HybridRetriever, tokenize

CASES = [
    ("How do I reserve a spot for an event?", "How to Reserve a Spot for an Event"),
    ("I forgot my password and can't log in", "How to Handle Login Issues?"),
    ("What's included in my subscription?", "What's Included in a CultPass Subscription"),
    ("How can I pause my subscription?", "How to Cancel or Pause a Subscription"),
    ("Can I bring a friend with me?", "Bringing a Guest to an Experience"),
    ("The QR code is not showing in the app", "App Troubleshooting: Crashes and QR Code Not Showing"),
    ("I was charged twice and want my money back", "Refund Requests and Billing Disputes"),
    ("Why is my account blocked?", "Account Blocked or Suspended"),
    ("Do unused experiences roll over to next month?", "Monthly Quota and Unused Experiences"),
    ("Is there wheelchair access at events?", "Accessibility and Special Assistance"),
    ("How do I delete my account data?", "Data Privacy and Deleting Your Account"),
    ("The venue cancelled the event, do I get my credit back?", "Event Cancelled or Changed by the Venue"),
    ("How do I change my email address?", "Changing Account Email or Personal Details"),
    ("My card payment failed", "Updating Payment Method and Failed Payments"),
    ("Are premium experiences included for basic members?", "Premium Experiences and Subscription Tiers"),
    ("Can I cancel my reservation the day before?", "How to Cancel a Reservation (24-Hour Policy)"),
]


@pytest.mark.parametrize("query,expected", CASES)
def test_top_result_is_the_right_article(dbs, query, expected):
    result = udahub_tools.search_knowledge(query, k=3)
    top = result["results"][0]
    assert top["title"] == expected
    assert top["score"] >= MIN_RELEVANCE


@pytest.mark.parametrize("query", ["What is the capital of France?", "recommend a pizza recipe", "hello"])
def test_off_topic_queries_fall_below_threshold(dbs, query):
    result = udahub_tools.search_knowledge(query, k=3)
    assert result["results"][0]["score"] < MIN_RELEVANCE


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("I was charged") == tokenize("charge")
    assert "the" not in tokenize("the reservations")


class KeywordEmbeddings(Embeddings):
    """Tiny deterministic embedding: one dimension per vocabulary word."""

    vocab = ["money", "refund", "password", "reset", "login"]

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        return [float(w in text.lower()) for w in self.vocab]


def test_hybrid_retriever_with_embeddings():
    docs = [Document("1", "Refunds", "money back policy"), Document("2", "Login", "password reset")]
    retriever = HybridRetriever(docs, embeddings=KeywordEmbeddings())
    ranked = retriever.search("I forgot my login", k=2)  # no TF-IDF overlap with "password reset"...
    assert ranked[0][0].doc_id == "2"                     # ...but the embedding similarity finds it
    assert ranked[0][1] > 0 and ranked[1][1] == 0
