from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel
from src.schema import ComplianceAnswer
from src.logger import get_logger
import os

logger = get_logger(__name__)

groq_api_key = os.environ.get("GROQ_API_KEY")
MODEL_NAME = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


def build_llm() -> BaseChatModel | None:
    """
    Plain LLM — used for query contextualization (rewriting follow-up questions).
    Does not use structured output since we just need a plain text rewritten question.
    """
    try:
        llm = ChatGroq(
            model=MODEL_NAME,
            api_key=groq_api_key,
            temperature=0,
        )
        logger.info(f"LLM ready: {MODEL_NAME}")
        return llm
    except Exception as e:
        logger.error(f"Error building LLM: {e}")
        return None


def build_structured_llm() -> BaseChatModel | None:
    """
    Structured LLM — used for the final answer generation.
    Bound to ComplianceAnswer schema so every response is a
    predictable, parseable Pydantic object.
    """
    try:
        llm = ChatGroq(
            model=MODEL_NAME,
            api_key=groq_api_key,
            temperature=0,
        )
        structured_llm = llm.with_structured_output(ComplianceAnswer)
        logger.info(f"Structured LLM ready: {MODEL_NAME}")
        return structured_llm
    except Exception as e:
        logger.error(f"Error building structured LLM: {e}")
        return None
