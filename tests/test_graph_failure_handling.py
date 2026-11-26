from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.graph import _node_extract_and_store, _node_gate_and_dedupe
from src.schemas import RawArticle


class _FakeDatabase:
    def __init__(self):
        self.seen: set[str] = set()
        self.events = []

    def is_url_seen(self, url: str) -> bool:
        return url in self.seen

    def mark_url_seen(self, url: str) -> None:
        self.seen.add(url)

    def get_recent_candidate_embeddings(self, days_back: int):
        return []

    def get_aranca_context_summary(self, max_chars: int) -> str:
        return ""

    def insert_event(self, event, canonical_url: str, embedding):
        self.events.append(event)
        self.mark_url_seen(canonical_url)
        return len(self.events)


class _FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)

    def generate_structured(self, **kwargs):
        return next(self.responses)


def _article() -> RawArticle:
    return RawArticle(
        company="Example Advisory",
        region="US",
        title="Example Advisory launches a new strategy practice",
        snippet="The firm announced a new strategy practice.",
        source_url="https://example.com/news?utm_source=feed",
        published_date="2026-11-25",
        source_type="rss",
    )


def _state(article: RawArticle) -> dict:
    return {"run_id": 1, "raw_articles": [article], "gated_articles": []}


class GraphFailureHandlingTests(unittest.TestCase):
    def test_gate_failure_does_not_mark_url_seen(self):
        article = _article()
        db = _FakeDatabase()
        llm = _FakeLLM([None])

        with patch("src.graph.embed_texts", return_value=[[]]):
            result = _node_gate_and_dedupe(_state(article), db, llm)

        self.assertNotIn("https://example.com/news", db.seen)
        self.assertEqual(result["gated_articles"], [(article, 3, False)])

    def test_explicit_irrelevant_decision_marks_url_seen(self):
        article = _article()
        db = _FakeDatabase()
        decision = SimpleNamespace(
            index=0,
            is_relevant=False,
            materiality_score=1,
            reason="No material business update",
        )
        llm = _FakeLLM([SimpleNamespace(decisions=[decision])])

        result = _node_gate_and_dedupe(_state(article), db, llm)

        self.assertIn("https://example.com/news", db.seen)
        self.assertEqual(result["gated_articles"], [])

    def test_extraction_failure_leaves_url_eligible_for_retry(self):
        article = _article()
        db = _FakeDatabase()
        state = {
            "run_id": 1,
            "gated_articles": [(article, 3, False)],
        }

        result = _node_extract_and_store(state, db, _FakeLLM([None]))

        self.assertNotIn("https://example.com/news", db.seen)
        self.assertEqual(result["inserted_count"], 0)
        self.assertEqual(db.events, [])

    def test_gate_failure_with_successful_extraction_requires_review(self):
        article = _article()
        db = _FakeDatabase()
        state = {
            "run_id": 1,
            "gated_articles": [(article, 3, False)],
        }
        analysis = SimpleNamespace(
            aranca_segment="Business Transformation & PMO Advisory",
            tracking_pillar="Business Updates",
            executive_summary="Example Advisory launched a strategy practice.",
            strategic_implication="This may increase competition for strategy work.",
            confidence=0.99,
        )

        with patch("src.graph.embed_single", return_value=[]):
            result = _node_extract_and_store(state, db, _FakeLLM([analysis]))

        self.assertEqual(result["inserted_count"], 1)
        self.assertEqual(db.events[0].review_status, "PENDING_REVIEW")
        self.assertIn("https://example.com/news", db.seen)

    def test_successful_gate_and_extraction_preserve_auto_approval(self):
        article = _article()
        db = _FakeDatabase()
        state = {
            "run_id": 1,
            "gated_articles": [(article, 4, True)],
        }
        analysis = SimpleNamespace(
            aranca_segment="Business Transformation & PMO Advisory",
            tracking_pillar="Business Updates",
            executive_summary="Example Advisory launched a strategy practice.",
            strategic_implication="This may increase competition for strategy work.",
            confidence=0.99,
        )

        with patch("src.graph.embed_single", return_value=[]):
            result = _node_extract_and_store(state, db, _FakeLLM([analysis]))

        self.assertEqual(result["inserted_count"], 1)
        self.assertEqual(db.events[0].review_status, "APPROVED")
        self.assertIn("https://example.com/news", db.seen)


if __name__ == "__main__":
    unittest.main()
