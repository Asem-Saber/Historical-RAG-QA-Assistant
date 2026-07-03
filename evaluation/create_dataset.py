"""Upload the evaluation dataset to LangSmith."""

import json
from pathlib import Path

from langsmith import Client

from app.core.config import settings


def main():
    eval_data_path = Path(settings.data_dir) / "eval_dataset.json"

    with open(eval_data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    client = Client()

    dataset_name = "Ancient Egypt RAG Evaluation"
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Evaluation QA pairs extracted from the Encyclopedia of Ancient Egypt.",
    )

    for item in eval_data:
        client.create_example(
            inputs=item["inputs"],
            outputs=item["outputs"],
            dataset_id=dataset.id,
        )

    print(f"Created dataset '{dataset_name}' with {len(eval_data)} examples.")


if __name__ == "__main__":
    main()
