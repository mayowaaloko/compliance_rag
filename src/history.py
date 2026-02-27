import psycopg2
import psycopg2.extras
import os
from langchain_core.messages import HumanMessage, AIMessage
from src.logger import get_logger

logger = get_logger(__name__)


DATABASE_URL = os.environ.get("DATABASE_URL")


def get_conn():
    """Returns a new Postgres connection. Called fresh each time to avoid stale connections."""
    return psycopg2.connect(DATABASE_URL)


def init_db() -> None:
    """Creates the messages table if it does not already exist."""
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id        SERIAL PRIMARY KEY,
                session   TEXT        NOT NULL,
                role      TEXT        NOT NULL,
                content   TEXT        NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW()
            )
        """
        )
        # Index on session for fast history lookups
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session)
        """
        )
        conn.commit()
        cur.close()
        conn.close()
        logger.info("Postgres chat history DB initialized.")
    except Exception as e:
        logger.error(f"Error initializing Postgres DB: {e}")


def load_history(session_id: str) -> list:
    """Loads chat history for a session as a list of LangChain message objects."""
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT role, content FROM messages WHERE session = %s ORDER BY id",
            (session_id,),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        history = []
        for role, content in rows:
            if role == "human":
                history.append(HumanMessage(content=content))
            else:
                history.append(AIMessage(content=content))

        logger.info(f"Loaded {len(history)} messages for session '{session_id}'")
        return history

    except Exception as e:
        logger.error(f"Error loading history for session '{session_id}': {e}")
        return []


def save_message(session_id: str, role: str, content: str) -> None:
    """Saves a single message to Postgres. Role must be 'human' or 'ai'."""
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO messages (session, role, content) VALUES (%s, %s, %s)",
            (session_id, role, content),
        )
        conn.commit()
        cur.close()
        conn.close()
        logger.debug(f"Saved [{role}] message for session '{session_id}'")
    except Exception as e:
        logger.error(f"Error saving message for session '{session_id}': {e}")


def clear_session(session_id: str) -> None:
    """Deletes all messages for a session. Useful for resetting a conversation."""
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM messages WHERE session = %s", (session_id,))
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Cleared history for session '{session_id}'")
    except Exception as e:
        logger.error(f"Error clearing session '{session_id}': {e}")
