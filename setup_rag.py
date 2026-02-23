"""One-time script to index RAG knowledge base documents into ChromaDB."""

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("setup_rag")


def main():
    logger.info("Initializing RAG knowledge base...")

    try:
        from core.rag_client import RAGClient
    except ImportError as e:
        logger.error("Failed to import RAGClient: %s", e)
        logger.info("Install chromadb: pip install chromadb")
        sys.exit(1)

    rag = RAGClient(persist_dir="./chroma_db")

    docs_dir = os.path.join(os.path.dirname(__file__), "rag_docs")
    if not os.path.isdir(docs_dir):
        logger.error("rag_docs/ directory not found at %s", docs_dir)
        sys.exit(1)

    md_files = [f for f in os.listdir(docs_dir) if f.endswith(".md")]
    logger.info("Found %d markdown files in %s", len(md_files), docs_dir)

    rag.index_documents(docs_dir)

    if rag.is_indexed():
        logger.info("RAG knowledge base indexed successfully!")
        results = rag.retrieve("How to deploy a model?", top_k=2)
        logger.info("Test query returned %d results", len(results))
        for i, r in enumerate(results):
            logger.info("Result %d: %s...", i + 1, r[:100])
    else:
        logger.warning("RAG indexing may have failed - no documents found")

    logger.info("Setup complete!")


if __name__ == "__main__":
    main()
