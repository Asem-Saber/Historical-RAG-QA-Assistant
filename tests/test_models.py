"""Tests for app.core.models — Pydantic schemas for structured LLM outputs."""

import pytest
from pydantic import ValidationError

from app.core.models import (
    AnswerRelevanceCheck,
    GradeDocuments,
    GradingResult,
    GuardrailScoring,
    HallucinationCheck,
)


class TestGuardrailScoring:
    def test_valid_score(self):
        g = GuardrailScoring(score=75, reason="Related to Egypt")
        assert g.score == 75
        assert g.reason == "Related to Egypt"

    def test_score_at_boundaries(self):
        assert GuardrailScoring(score=0, reason="not relevant").score == 0
        assert GuardrailScoring(score=100, reason="very relevant").score == 100

    def test_score_below_zero_raises(self):
        with pytest.raises(ValidationError):
            GuardrailScoring(score=-1, reason="invalid")

    def test_score_above_100_raises(self):
        with pytest.raises(ValidationError):
            GuardrailScoring(score=101, reason="invalid")


class TestGradeDocuments:
    def test_valid_yes(self):
        g = GradeDocuments(binary_score="yes", reasoning="Relevant")
        assert g.binary_score == "yes"

    def test_valid_no(self):
        g = GradeDocuments(binary_score="no", reasoning="Not relevant")
        assert g.binary_score == "no"

    def test_invalid_score_raises(self):
        with pytest.raises(ValidationError):
            GradeDocuments(binary_score="maybe", reasoning="unsure")

    def test_reasoning_defaults_to_empty(self):
        g = GradeDocuments(binary_score="yes")
        assert g.reasoning == ""


class TestGradingResult:
    def test_full_construction(self):
        r = GradingResult(
            document_id="doc_1",
            is_relevant=True,
            score=0.85,
            reasoning="Matches topic",
        )
        assert r.document_id == "doc_1"
        assert r.is_relevant is True
        assert r.score == 0.85

    def test_defaults(self):
        r = GradingResult(document_id="x", is_relevant=False)
        assert r.score == 0.0
        assert r.reasoning == ""


class TestHallucinationCheck:
    def test_grounded_yes(self):
        h = HallucinationCheck(grounded="yes", reasoning="All facts match")
        assert h.grounded == "yes"

    def test_grounded_no(self):
        h = HallucinationCheck(grounded="no", reasoning="Invented a date")
        assert h.grounded == "no"

    def test_invalid_raises(self):
        with pytest.raises(ValidationError):
            HallucinationCheck(grounded="partially", reasoning="some match")


class TestAnswerRelevanceCheck:
    def test_relevant_yes(self):
        a = AnswerRelevanceCheck(relevant="yes", reasoning="Addresses the question")
        assert a.relevant == "yes"

    def test_relevant_no(self):
        a = AnswerRelevanceCheck(relevant="no", reasoning="Off topic")
        assert a.relevant == "no"

    def test_invalid_raises(self):
        with pytest.raises(ValidationError):
            AnswerRelevanceCheck(relevant="somewhat", reasoning="partial")
