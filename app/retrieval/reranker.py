from sentence_transformers import CrossEncoder
from langsmith import traceable


class Reranker:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model = CrossEncoder(model_name, device=device)

    @traceable(name="Re-rank Documents", run_type="retriever")
    def rerank(self, query: str, docs, top_n: int = 5):
        if not docs:
            return docs
        scores = self.model.rank(query, [doc.page_content for doc in docs], top_k=top_n)
        return [docs[result["corpus_id"]] for result in scores]
