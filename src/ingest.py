# -------- INGESTION, CLEANING, CHUNKING, QDRANT PIPELINES ---------

from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import os
import fitz
import re
from src.logger import get_logger

logger = get_logger(__name__)

load_dotenv()

qdrant_url = os.environ.get("QDRANT_URL")
qdrant_api_key = os.environ.get("QDRANT_API_KEY")
qdrant_collection_name = os.environ.get("QDRANT_COLLECTION_NAME")
openai_api_key = os.environ.get("OPENAI_API_KEY")
LAW_CATEGORY_MAP = {
    "Taxation": ["tax", "nta", "revenue", "firs", "nrs"],
    "Data Protection": ["data", "ndpa", "privacy", "ndpc"],
    "Corporate/CAC": ["cama", "cac", "corporate", "annual return"],
    "Employment": ["labour", "pension", "wage", "employee"],
}


def load_documents(path: Path) -> List[Document]:
    """Extract text page-by-page from a PDF."""
    docs = []
    logger.info(f"Loading documents from {path}")
    try:
        with fitz.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text().strip()
                if text:  # skip blank pages
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(path.name),
                                "file_type": str(path),
                                "total_pages": pdf.page_count,
                                "page": page_num,
                            },
                        )
                    )
        logger.info(f"Loaded {len(docs)} pages from {path}")
        return docs
    except Exception as e:
        logger.error(f"Error loading document {path}: {e}")
        return []


def clean_documents(docs: List[Document]) -> List[Document]:
    docs = []
    for doc in docs:
        cleaned_content = re.sub(
            r"[ \t]+",
            " ",
            re.sub(r"\n{3,}", "\n\n", re.sub(r"\r\n?", "\n", doc.page_content)),
        ).strip()
        if cleaned_content:
            docs.append(Document(page_content=cleaned_content, metadata=doc.metadata))
    return docs


def tag_law_category(source_name: str) -> str:
    """Infer law category from the source file name."""
    source_lower = source_name.lower()
    for category, keywords in LAW_CATEGORY_MAP.items():
        if any(word in source_lower for word in keywords):
            return category
    return "General Compliance"


def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Splits documents into chunks and tags each chunk with law category.
    Markdown-aware separators preserve table structure from pymupdf4llm.
    """
    try:
        logger.info(f"Chunking {len(docs)} pages...")

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=900,
            chunk_overlap=150,
            separators=[
                "\nPART ",
                "\nPart ",
                "\nCHAPTER ",
                "\nChapter ",
                "\nSECTION ",
                "\nSection ",
                "\nArticle ",
                "\nARTICLE ",
                "\nSCHEDULE ",
                "\nSchedule ",
                "\n\n",
                "\n(",
                "\n(a)",
                "\n(i)",
                "\n• ",
                "\n- ",
                "\n",
                ". ",
                " ",
                "",
            ],
            add_start_index=True,
        )

        chunks = splitter.split_documents(docs)
        logger.info(f"Created {len(chunks)} chunks before tagging")

        # ---- METADATA TAGGING -----------------------------------
        for chunk in chunks:
            source_name = chunk.metadata.get("source", "")

            # Tag law category (overwrite the "General" placeholder from loader)
            chunk.metadata["law_category"] = tag_law_category(source_name)

            # Flag chunks that contain markdown tables — helps LLM and retrieval
            if "|" in chunk.page_content and "---" in chunk.page_content:
                chunk.metadata["contains_table"] = True
            else:
                chunk.metadata["contains_table"] = False

        logger.info(f"Tagged {len(chunks)} chunks with law category and table flags")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking documents: {e}")
        return []


def qdrant_ingest(chunks: List[Document]) -> QdrantVectorStore | None:
    """
    Ingests chunks into Qdrant.
    - If the collection already exists, skips ingestion to avoid re-embedding.
    - If the collection does not exist, creates it and ingests all chunks.
    """
    try:
        logger.info(f"Connecting to Qdrant at {qdrant_url}...")
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

        dense_embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", api_key=openai_api_key
        )
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        # ---- CHECK IF COLLECTION EXISTS ---------------------
        existing_collections = [c.name for c in client.get_collections().collections]

        if qdrant_collection_name in existing_collections:
            logger.info(
                f"Collection '{qdrant_collection_name}' already exists with "
                f"{client.count(qdrant_collection_name).count} vectors. "
                f"Skipping ingestion — connecting to existing collection."
            )
            # Just connect to the existing collection, don't re-ingest
            qdrant_store = QdrantVectorStore(
                client=client,
                collection_name=qdrant_collection_name,
                embedding=dense_embeddings,
                sparse_embedding=sparse_embeddings,
                retrieval_mode=RetrievalMode.HYBRID,
            )
            return qdrant_store

        # ---- INGEST IF COLLECTION DOES NOT EXIST ------------
        logger.info(
            f"Collection not found. Ingesting {len(chunks)} chunks into Qdrant..."
        )
        qdrant_store = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            collection_name=qdrant_collection_name,
            url=qdrant_url,
            api_key=qdrant_api_key,
            retrieval_mode=RetrievalMode.HYBRID,
        )
        logger.info(
            f"Successfully ingested {len(chunks)} chunks into '{qdrant_collection_name}'."
        )
        return qdrant_store

    except Exception as e:
        logger.error(f"Error during Qdrant ingestion: {e}")
        return None
