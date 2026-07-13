"""Pydantic schemas for structured LLM outputs."""

from typing import Literal
from pydantic import BaseModel, Field


class GuardrailScoring(BaseModel):
    """Scoring result for guardrail validation."""

    score: int = Field(ge=0, le=100, description="Relevance score between 0 and 100")
    reason: str = Field(description="Brief reason for the score")


class GradeDocuments(BaseModel):
    """Binary score for document relevance check."""

    binary_score: Literal["yes", "no"] = Field(description="Document relevance: 'yes' or 'no'")
    reasoning: str = Field(default="", description="Explanation for the decision")


class GradingResult(BaseModel):
    """Result of document grading with details."""

    document_id: str = Field(description="Document identifier")
    is_relevant: bool = Field(description="Relevance flag")
    score: float = Field(default=0.0, description="Relevance score")
    reasoning: str = Field(default="", description="Grading reasoning")


class HallucinationCheck(BaseModel):
    """Binary check for answer grounding in sources."""

    grounded: Literal["yes", "no"] = Field(description="Is answer grounded in sources: 'yes' or 'no'")
    reasoning: str = Field(default="", description="Explanation for the decision")


class AnswerRelevanceCheck(BaseModel):
    """Binary check for answer relevance to the question."""

    relevant: Literal["yes", "no"] = Field(description="Does answer address the question: 'yes' or 'no'")
    reasoning: str = Field(default="", description="Explanation for the decision")
