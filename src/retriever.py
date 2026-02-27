from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_cohere import CohereRerank
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from src.logger import get_logger
import os

logger = get_logger(__name__)

cohere_api_key = os.environ.get("COHERE_API_KEY")

TOP_K = 6  # chunks retrieved from Qdrant before reranking
TOP_N = 3  # chunks kept after Cohere reranking


def build_retriever(
    qdrant_store: QdrantVectorStore,
) -> ContextualCompressionRetriever | None:
    """
    Builds the retriever pipeline:
    - Base retriever: hybrid search (dense + sparse) from Qdrant
    - Reranker: Cohere rerank narrows TOP_K down to TOP_N most relevant chunks
    """

    try:
        logger.info("Building base retriever...")
        base_retriever = qdrant_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K, "fetch_k": 20, "lambda_mult": 0.5},
        )

        logger.info("Attaching Cohere reranker...")
        reranker = CohereRerank(top_n=TOP_N, model="rerank-english-v3.0")

        logger.info("Building contextual compression retriever...")
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=base_retriever
        )
        logger.info("Successfully built retriever pipeline.")
        return compression_retriever
    except Exception as e:
        logger.error(f"Error building retriever pipeline: {e}")
        return None
