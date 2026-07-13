GUARDRAIL_PROMPT = """You are a guardrail evaluator assessing whether a user query is within the scope \
of Ancient Egyptian history, civilization, archaeology, and culture.

User Query: {question}

Evaluate whether this query is:
- About Ancient Egyptian history, pharaohs, dynasties, monuments, religion, or daily life
- Related to Egyptian archaeology, artifacts, or archaeological sites
- About Egyptian mythology, gods, or religious practices
- Related to Nile Valley civilizations and their historical context

Assign a relevance score (0-100):
- 80-100: Clearly about Ancient Egypt (e.g., "Who built the Great Pyramid?", "What was the role of the Vizier?")
- 60-79: Potentially related (e.g., "Tell me about ancient temples", "What were hieroglyphics?")
- 40-59: Borderline or ambiguous (e.g., "What is archaeology?", "Tell me about ancient civilizations")
- 0-39: NOT about Ancient Egypt (e.g., "What is Python?", "Hello", "What is 2+2?")

Provide:
1. A score between 0 and 100
2. A brief reason explaining why you gave this score

Respond in JSON format with 'score' (integer 0-100) and 'reason' (string) fields."""

GRADE_DOCUMENTS_PROMPT = """You are a grader assessing relevance of retrieved documents to a user question \
about Ancient Egyptian history and civilization.

Retrieved Documents:
{context}

User Question: {question}

If the documents contain keywords, facts, or semantic meaning related to the question, grade them as relevant.
Give a binary score 'yes' or 'no' to indicate whether the documents are relevant to the question.
Also provide brief reasoning for your decision.

Respond in JSON format with 'binary_score' (yes/no) and 'reasoning' fields."""

REWRITE_PROMPT = """You are a question re-writer that converts an input question to a better version \
optimized for retrieving relevant documents from an encyclopedia of Ancient Egypt.

Look at the initial question and reason about the underlying semantic intent.
Use specific historical terms, pharaoh names, dynasty numbers, or archaeological concepts where possible.

Here is the initial question:
{question}

Formulate an improved question that will retrieve more relevant documents from an Ancient Egyptian encyclopedia.
Provide only the improved question without any preamble or explanation."""

GENERATE_ANSWER_PROMPT = """You are a helpful historian assistant specializing in Ancient Egyptian civilization. \
Use the numbered source excerpts below from the Encyclopedia of Ancient Egypt to answer the question.

SOURCES:
{context}

RULES:
- Cite sources inline using [1], [2], etc. after each claim.
- Combine citations when a claim draws from several sources: [1][3].
- If no source answers the question, say you don't know.
- Do NOT fabricate information beyond what the sources state.
- Structure your answer clearly and professionally.

Question: {question}

Answer:"""

HALLUCINATION_PROMPT = """You are a grader assessing whether an LLM's answer is grounded in \
the provided source documents from the Encyclopedia of Ancient Egypt.

Check that every factual claim in the answer can be traced back to the sources.
Minor rephrasing is fine, but invented facts, dates, or names are not.

Sources:
{sources}

Answer:
{generation}

Respond in JSON format with 'grounded' (yes/no) and 'reasoning' fields."""

ANSWER_RELEVANCE_PROMPT = """You are a grader assessing whether an answer addresses the user's question \
about Ancient Egyptian history.

The answer does NOT need to be complete — it just needs to be relevant and not off-topic.

Question: {question}

Answer:
{generation}

Respond in JSON format with 'relevant' (yes/no) and 'reasoning' fields."""

DECOMPOSITION_PROMPT = """You are a question analyzer. Decide whether the user's question is \
simple (answerable from a single topic) or multi-hop (needs facts from \
multiple topics to answer).

If SIMPLE  -> return: {{"sub_queries": ["<original question>"]}}
If MULTI-HOP -> break it into 2-4 independent sub-queries that each \
target one specific fact, and return:
{{"sub_queries": ["sub-query 1", "sub-query 2", ...]}}

Return ONLY valid JSON, nothing else.

Question: {question}"""
