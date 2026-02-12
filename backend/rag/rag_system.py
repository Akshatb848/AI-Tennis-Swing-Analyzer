"""
Lightweight RAG (Retrieval-Augmented Generation) system.

Uses TF-IDF + cosine similarity via numpy for zero-hallucination
document retrieval from the RVSK newsletter ground-truth data.
"""

import json
import math
import logging
import pathlib
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("moe_rag")

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "newsletter_data.json"


class Document:
    """A single searchable text chunk with metadata."""

    __slots__ = ("text", "metadata", "month", "section")

    def __init__(self, text: str, metadata: Dict[str, Any], month: str, section: str):
        self.text = text
        self.metadata = metadata
        self.month = month
        self.section = section


class RAGSystem:
    """TF-IDF vector search over newsletter documents."""

    def __init__(self):
        self.documents: List[Document] = []
        self.vocab: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self._raw_data: Dict[str, Any] = {}
        self._indexed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_and_index(self, data_path: Optional[str] = None) -> int:
        """Load newsletter JSON and build TF-IDF index. Returns doc count."""
        path = pathlib.Path(data_path) if data_path else DATA_PATH
        with open(path, "r") as f:
            raw = json.load(f)
        self._raw_data = raw
        self.documents = self._build_documents(raw)
        self._build_index()
        self._indexed = True
        logger.info(f"RAG indexed {len(self.documents)} documents")
        return len(self.documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top-k relevant documents for a query."""
        if not self._indexed or not self.documents:
            return []
        q_vec = self._query_vector(query)
        if q_vec is None:
            return []
        scores = self.tfidf_matrix.dot(q_vec)
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            if scores[idx] <= 0:
                break
            doc = self.documents[idx]
            results.append({
                "text": doc.text,
                "month": doc.month,
                "section": doc.section,
                "relevance": round(float(scores[idx]), 4),
                "metadata": doc.metadata,
            })
        return results

    def get_month_data(self, month: str) -> Optional[Dict[str, Any]]:
        """Direct lookup for a specific month's full data."""
        return self._raw_data.get("months", {}).get(month)

    def get_all_months(self) -> List[str]:
        """Return sorted list of available month keys."""
        return sorted(self._raw_data.get("months", {}).keys(),
                      key=self._month_sort_key)

    def get_apaar_trend(self) -> List[Dict[str, Any]]:
        """Return APAAR ID generation trend across months."""
        trend = []
        for month in self.get_all_months():
            mdata = self._raw_data["months"][month]
            total = mdata.get("apaar_ids_total")
            if total is not None:
                trend.append({
                    "month": month,
                    "total": total,
                    "date": mdata.get("apaar_ids_date", month),
                })
        return trend

    def compare_states(self, states: List[str], month: Optional[str] = None) -> Dict[str, Any]:
        """Search for data related to the given states."""
        query = " ".join(states)
        results = self.search(query, top_k=10)
        target_month = month or (self.get_all_months()[-1] if self.get_all_months() else "")
        return {"month": target_month, "states": {s: {} for s in states}, "results": results}

    def get_sections_for_month(self, month: str) -> Dict[str, List[str]]:
        """Return all section data for a specific month."""
        mdata = self.get_month_data(month)
        if not mdata:
            return {}
        return mdata.get("sections", {})

    # ------------------------------------------------------------------
    # Internal — Document Building
    # ------------------------------------------------------------------

    def _build_documents(self, raw: Dict[str, Any]) -> List[Document]:
        """Flatten newsletter JSON into searchable Document chunks."""
        docs = []
        for month, mdata in raw.get("months", {}).items():
            # Highlights
            for h in mdata.get("highlights", []):
                docs.append(Document(
                    text=h,
                    metadata={"type": "highlight", "theme": mdata.get("theme", "")},
                    month=month,
                    section="highlights",
                ))

            # Theme
            theme = mdata.get("theme", "")
            if theme:
                docs.append(Document(
                    text=f"Newsletter theme for {month}: {theme}",
                    metadata={"type": "theme"},
                    month=month,
                    section="theme",
                ))

            # APAAR IDs
            apaar_total = mdata.get("apaar_ids_total")
            if apaar_total is not None:
                apaar_date = mdata.get("apaar_ids_date", month)
                docs.append(Document(
                    text=f"APAAR ID Generation: {apaar_total:,} IDs generated as of {apaar_date}.",
                    metadata={"type": "apaar", "total": apaar_total, "date": apaar_date},
                    month=month,
                    section="apaar",
                ))

            # Sections
            for section_key, items in mdata.get("sections", {}).items():
                for item in items:
                    docs.append(Document(
                        text=item,
                        metadata={"type": "section", "section_key": section_key},
                        month=month,
                        section=section_key,
                    ))
        return docs

    # ------------------------------------------------------------------
    # Internal — TF-IDF
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenizer."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _build_index(self):
        """Build TF-IDF vectors for all documents."""
        all_tokens = [self._tokenize(d.text) for d in self.documents]

        vocab_set: Dict[str, int] = {}
        idx = 0
        for tokens in all_tokens:
            for t in tokens:
                if t not in vocab_set:
                    vocab_set[t] = idx
                    idx += 1
        self.vocab = vocab_set
        V = len(vocab_set)
        N = len(self.documents)

        if V == 0 or N == 0:
            self.tfidf_matrix = np.zeros((0, 0))
            self.idf = np.zeros(0)
            return

        df = np.zeros(V)
        for tokens in all_tokens:
            seen = set(tokens)
            for t in seen:
                df[self.vocab[t]] += 1

        self.idf = np.log((N + 1) / (df + 1)) + 1

        mat = np.zeros((N, V))
        for i, tokens in enumerate(all_tokens):
            tf = Counter(tokens)
            for t, count in tf.items():
                j = self.vocab[t]
                mat[i, j] = (1 + math.log(count)) * self.idf[j]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.tfidf_matrix = mat / norms

    def _query_vector(self, query: str) -> Optional[np.ndarray]:
        """Convert query string to TF-IDF vector."""
        tokens = self._tokenize(query)
        if not tokens:
            return None
        V = len(self.vocab)
        vec = np.zeros(V)
        tf = Counter(tokens)
        for t, count in tf.items():
            if t in self.vocab:
                j = self.vocab[t]
                vec[j] = (1 + math.log(count)) * self.idf[j]
        norm = np.linalg.norm(vec)
        if norm == 0:
            return None
        return vec / norm

    @staticmethod
    def _month_sort_key(month_str: str) -> Tuple[int, int]:
        """Sort key: (year, month_num)."""
        month_names = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }
        parts = month_str.strip().split()
        if len(parts) == 2:
            m = month_names.get(parts[0].lower(), 0)
            try:
                y = int(parts[1])
            except ValueError:
                y = 0
            return (y, m)
        return (0, 0)
