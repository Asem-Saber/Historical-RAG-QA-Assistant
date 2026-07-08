"""Run LangSmith evaluation experiments on the RAG pipeline."""

from langsmith import Client
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.pipeline import AncientEgyptRAG


DATASET_NAME = "Ancient Egypt RAG Evaluation"

judge_llm = ChatOpenAI(
    base_url=settings.llm_endpoint,
    api_key=settings.api_key,
    model="openai/gpt-4o-mini",
    temperature=0,
)


def predict_rag_answer(inputs: dict) -> dict:
    """Run the RAG pipeline on a single question."""
    question = inputs["question"]
    response = AncientEgyptRAG(question)
    return {"predicted_answer": response["answer"]}


def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """Evaluate answer correctness using an LLM judge."""
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        judge=judge_llm,
        feedback_key="correctness",
    )
    return evaluator(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=reference_outputs,
    )


def main():
    client = Client()
    experiment_results = client.evaluate(
        predict_rag_answer,
        data=DATASET_NAME,
        evaluators=[correctness_evaluator],
        experiment_prefix="ancient-egypt-baseline",
        metadata={
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "parser": "LlamaParse-Agentic",
        },
        max_concurrency=2,
    )
    print(experiment_results)


if __name__ == "__main__":
    main()
