"""ChromaDB-based RAG (Retrieval-Augmented Generation) client.

Indexes markdown documentation into a persistent ChromaDB collection and
provides semantic search for retrieval. Uses the default embedding function
so no external API key is required.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from config.settings import LOG_FORMAT

logger = logging.getLogger("rag_client")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
COLLECTION_NAME = "knowledge_base"


def _split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks by character count.

    Args:
        text: The source text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        if end < text_len:
            for sep in ["\n\n", "\n", ". ", " "]:
                last_sep = text.rfind(sep, start, end)
                if last_sep > start:
                    end = last_sep + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end < text_len else text_len

    return chunks


class RAGClient:
    """Retrieval-Augmented Generation client backed by ChromaDB.

    Attributes:
        persist_dir: Directory where ChromaDB stores its data.
    """

    def __init__(self, persist_dir: str = "./chroma_db") -> None:
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None
        self._init_error: Optional[str] = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the ChromaDB client and collection."""
        try:
            import chromadb

            os.makedirs(self.persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB initialised (dir=%s, docs=%d)",
                self.persist_dir,
                self._collection.count(),
            )
        except ImportError:
            self._init_error = "chromadb package is not installed"
            logger.error("ChromaDB not available: %s", self._init_error)
        except Exception as exc:
            self._init_error = str(exc)
            logger.error("Failed to initialise ChromaDB: %s", exc)

    @property
    def is_ready(self) -> bool:
        """Return True if the ChromaDB client is properly initialised."""
        return self._collection is not None

    def index_documents(self, docs_dir: str) -> int:
        """Load markdown files from a directory, chunk them, and index in ChromaDB.

        Args:
            docs_dir: Path to directory containing ``.md`` files.

        Returns:
            Number of chunks indexed, or 0 on failure.
        """
        if not self.is_ready:
            logger.error("Cannot index: ChromaDB not initialised (%s)", self._init_error)
            return 0

        docs_path = Path(docs_dir)
        if not docs_path.exists() or not docs_path.is_dir():
            logger.warning("Documents directory does not exist: %s", docs_dir)
            return 0

        md_files = list(docs_path.glob("**/*.md"))
        if not md_files:
            logger.info("No .md files found in %s", docs_dir)
            return 0

        all_chunks: List[str] = []
        all_ids: List[str] = []
        all_metadata: List[dict] = []

        for md_file in md_files:
            try:
                content = md_file.read_text(encoding="utf-8", errors="replace")
                chunks = _split_text(content)
                for i, chunk in enumerate(chunks):
                    doc_id = f"{md_file.stem}_chunk_{i}"
                    all_chunks.append(chunk)
                    all_ids.append(doc_id)
                    all_metadata.append({
                        "source": str(md_file.relative_to(docs_path)),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    })
            except Exception as exc:
                logger.warning("Failed to read %s: %s", md_file, exc)
                continue

        if not all_chunks:
            logger.info("No content to index after processing %d files", len(md_files))
            return 0

        batch_size = 100
        indexed = 0
        for start in range(0, len(all_chunks), batch_size):
            end = min(start + batch_size, len(all_chunks))
            try:
                self._collection.upsert(
                    ids=all_ids[start:end],
                    documents=all_chunks[start:end],
                    metadatas=all_metadata[start:end],
                )
                indexed += end - start
            except Exception as exc:
                logger.error("Failed to index batch %d-%d: %s", start, end, exc)

        logger.info("Indexed %d chunks from %d files into '%s'", indexed, len(md_files), COLLECTION_NAME)
        return indexed

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Perform semantic search over the indexed documents.

        Args:
            query: Natural-language search query.
            top_k: Maximum number of results to return.

        Returns:
            List of matching text chunks, ordered by relevance.
        """
        if not self.is_ready:
            logger.error("Cannot retrieve: ChromaDB not initialised (%s)", self._init_error)
            return []

        if not query or not query.strip():
            return []

        try:
            count = self._collection.count()
            if count == 0:
                logger.info("Collection is empty — nothing to retrieve")
                return []

            effective_k = min(top_k, count)
            results = self._collection.query(
                query_texts=[query],
                n_results=effective_k,
            )

            documents = results.get("documents", [[]])[0]
            logger.info("Retrieved %d chunks for query: '%s'", len(documents), query[:80])
            return documents
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            return []

    def is_indexed(self) -> bool:
        """Check whether the collection contains any indexed documents.

        Returns:
            True if the collection has at least one document.
        """
        if not self.is_ready:
            return False
        try:
            return self._collection.count() > 0
        except Exception as exc:
            logger.error("Failed to check index status: %s", exc)
            return False
