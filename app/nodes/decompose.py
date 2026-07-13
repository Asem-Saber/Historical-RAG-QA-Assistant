import logging

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.prompts import DECOMPOSITION_PROMPT
from app.core.state import AgentState
from app.nodes.utils import get_latest_query

logger = logging.getLogger(__name__)

prompt_decomposition = ChatPromptTemplate.from_template(DECOMPOSITION_PROMPT)


def decompose_query_step(state: AgentState, llm: ChatOpenAI) -> dict:
    """Decompose the query into sub-queries for multi-hop retrieval."""
    logger.info("NODE: decompose_query")

    query = get_latest_query(state["messages"])

    try:
        chain = prompt_decomposition | llm | JsonOutputParser()
        result = chain.invoke({"question": query})
        sub_queries = result.get("sub_queries", [query])
        if not sub_queries:
            sub_queries = [query]

        logger.info(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")

    except Exception as e:
        logger.warning(f"Decomposition failed: {e}, using original query")
        sub_queries = [query]

    return {"sub_queries": sub_queries}
