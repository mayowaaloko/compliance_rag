from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

from src.logger import get_logger
from src.history import init_db
from src.ingest import (
    load_documents,
    chunk_documents,
)
from src.ingest import qdrant_ingest
from src.retriever import build_retriever
from src.llm import build_llm, build_structured_llm
from src.rag import rag_query

logger = get_logger(__name__)


# ============================================================
# STARTUP — runs once when the app starts
# ============================================================

# 1. Init Postgres
logger.info("Initializing Postgres...")
init_db()

# 2. Load documents
logger.info("Loading documents...")
documents_path = Path("documents")
doc = []
for pdf_file in documents_path.glob("*.pdf"):
    doc.extend(load_documents(pdf_file))
logger.info(f"Loaded {len(doc)} pages total")

# 3. Chunk + tag
logger.info("Chunking documents...")
chunks = chunk_documents(doc)

# 4. Ingest into Qdrant (skips if collection already exists)
logger.info("Setting up Qdrant...")
qdrant_store = qdrant_ingest(chunks)
if qdrant_store is None:
    logger.error("Qdrant setup failed. Exiting.")
    raise RuntimeError("Could not connect to or ingest into Qdrant.")

# 5. Build retriever
logger.info("Building retriever...")
retriever = build_retriever(qdrant_store)
if retriever is None:
    logger.error("Retriever setup failed. Exiting.")
    raise RuntimeError("Could not build retriever.")

# 6. Build LLMs
logger.info("Building LLMs...")
llm = build_llm()
structured_llm = build_structured_llm()
if llm is None or structured_llm is None:
    logger.error("LLM setup failed. Exiting.")
    raise RuntimeError("Could not build LLMs.")

logger.info("All systems ready.")


# ============================================================
# QUERY — call this per user request
# ============================================================
def query(question: str, session_id: str):
    """
    Entry point for a single user query.
    Pass in the question and a unique session_id per user/conversation.
    Returns a ComplianceAnswer object.
    """
    logger.info(f"New query | session='{session_id}' | question='{question}'")
    result = rag_query(
        question=question,
        session_id=session_id,
        retriever=retriever,
        llm=llm,
        structured_llm=structured_llm,
    )
    return result


# ============================================================
# ENTRY POINT — for local testing only
# ============================================================
if __name__ == "__main__":
    print("\nCompliance RAG ready. Type 'quit' to exit.\n")
    session_id = "local_test_session"

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            break

        result = query(question, session_id=session_id)

        if result:
            print(f"\nAssistant: {result.answer}")
            print(f"Found in docs: {result.found_in_docs}")
            print(f"Confidence: {result.confidence}")
            print("Sources:")
            for s in result.sources:
                print(
                    f"  - {s.document_name} | Page {s.page_number} | Section {s.section} | Category {s.law_category}"
                )
            print()
        else:
            print("\nSomething went wrong. Check the logs.\n")
