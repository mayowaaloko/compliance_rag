from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_core.language_models import BaseChatModel
from src.schema import ComplianceAnswer
from src.history import load_history, save_message
from src.logger import get_logger
from typing import Optional

logger = get_logger(__name__)


# ---- PROMPTS ------------------------------------------------
CONTEXTUALIZE_PROMPT = """Given the chat history and the latest user question, \
rewrite the question as a fully standalone question that can be understood \
without the chat history. Do NOT answer it, only rewrite it. \
If it is already standalone, return it as is."""

SYSTEM_PROMPT = """You are a precise compliance assistant for Nigerian regulatory and legal documents. \
Answer using ONLY the provided context from the compliance documents. \
Always cite the source document name, page number, and section if available. \
Sections prefixed with CONTAINED_TABLE_DATA contain structured table information — \
pay close attention to these when answering questions about figures, thresholds, or rates. \
Set found_in_docs to False and explain that the answer was not found if the context does not support an answer. \
Set confidence based on how clearly the context supports the answer: high, medium, or low. \
Never make up or infer information beyond what is in the context."""


# ---- HELPERS ------------------------------------------------
def format_context(docs: list[Document]) -> str:
    """Formats retrieved chunks into a numbered context string for the prompt."""
    parts = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "Unknown")
        page = d.metadata.get("page", "?")
        category = d.metadata.get("law_category", "General Compliance")
        parts.append(
            f"[{i}] Source: {source} | Page: {page} | Category: {category}\n{d.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def contextualize_question(
    question: str,
    history: list,
    llm: BaseChatModel,
) -> str:
    """
    Rewrites a follow-up question into a standalone question.
    Only calls the LLM if there is existing chat history.
    """
    if not history:
        return question

    messages = (
        [SystemMessage(content=CONTEXTUALIZE_PROMPT)]
        + history
        + [HumanMessage(content=question)]
    )
    response = llm.invoke(messages)
    standalone = response.content
    logger.debug(f"Contextualized question: {standalone}")
    return standalone


# ---- MAIN RAG FUNCTION --------------------------------------
def rag_query(
    question: str,
    session_id: str,
    retriever: ContextualCompressionRetriever,
    llm: BaseChatModel,
    structured_llm: BaseChatModel,
) -> Optional[ComplianceAnswer]:
    """
    Full RAG pipeline:
    1. Load chat history from SQLite
    2. Contextualize question
    3. Retrieve + rerank chunks
    4. Build prompt with context and history
    5. Get structured answer from LLM
    6. Save to SQLite
    7. Return ComplianceAnswer object
    """
    try:
        # 1. Load history
        history = load_history(session_id)

        # 2. Contextualize
        standalone_question = contextualize_question(question, history, llm)

        # 3. Retrieve + rerank
        logger.info(f"Retrieving chunks for: {standalone_question}")
        retrieved_docs = retriever.invoke(standalone_question)
        logger.info(f"Retrieved {len(retrieved_docs)} chunks after reranking")

        # 4. Build prompt
        context = format_context(retrieved_docs)
        messages = (
            [SystemMessage(content=SYSTEM_PROMPT)]
            + history
            + [HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")]
        )

        # 5. Get structured answer
        logger.info("Invoking structured LLM...")
        result: ComplianceAnswer = structured_llm.invoke(messages)

        # 6. Save to history
        save_message(session_id, "human", question)
        save_message(session_id, "ai", result.answer)

        logger.info(
            f"Answer generated. Found in docs: {result.found_in_docs} | Confidence: {result.confidence}"
        )
        return result

    except Exception as e:
        logger.error(f"Error in rag_query: {e}")
        return None
