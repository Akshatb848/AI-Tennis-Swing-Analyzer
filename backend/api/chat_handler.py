"""
Structured ChatHandler for the Education Intelligence Dashboard.

Returns deterministic, ground-truth-backed responses with:
  - intent classification
  - key_points extracted from newsletter data
  - sources with metadata
  - visualization hints for the frontend
  - confidence indicator

Zero-hallucination design: answers always come from RVSK newsletter data.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("moe_chat_handler")

# Intent keywords for rule-based classification
_INTENT_PATTERNS = {
    "apaar": re.compile(r"\b(apaar|id generation|ids generated)\b", re.I),
    "dashboard": re.compile(r"\b(dashboard|nipun|safal|nas|pm.?shri|ncert|attendance dashboard|assessment dashboard)\b", re.I),
    "vsk": re.compile(r"\b(vsks?|vidya samiksha|samiksha kendra|rvsk)\b", re.I),
    "state_query": re.compile(r"\b(kerala|bihar|west bengal|madhya pradesh|chhattisgarh|manipur|meghalaya|uttarakhand|goa|assam|himachal|chandigarh|delhi|andaman|ladakh|jharkhand|lakshadweep|telangana|sikkim|gujarat|tamil nadu|j&k)\b", re.I),
    "policy": re.compile(r"\b(nep|diksha|nishtha|swayam|samagra|naac|cuet|ncf|bharatnet|udise|lgd|ncte|nios|cbse|parakh|psscive)\b", re.I),
    "summary": re.compile(r"\b(summary|overview|highlight|theme|what happened|newsletter|month)\b", re.I),
    "meeting": re.compile(r"\b(meeting|workshop|session|conclave|visit|delegation)\b", re.I),
    "correspondence": re.compile(r"\b(letter|correspondence|request|proposal|reminder|communication)\b", re.I),
    "help": re.compile(r"\b(help|what can you|how do|capabilities)\b", re.I),
}


class ChatHandler:
    """Deterministic, data-backed chat response generator."""

    def __init__(self, rag_system):
        self.rag = rag_system

    def handle(self, query: str, current_month: Optional[str] = None) -> Dict[str, Any]:
        """Process a user query and return a structured response."""
        intent = self._classify_intent(query)
        handler = getattr(self, f"_handle_{intent}", self._handle_general)
        return handler(query, current_month, intent)

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    def _classify_intent(self, query: str) -> str:
        scores = {}
        for intent, pattern in _INTENT_PATTERNS.items():
            matches = pattern.findall(query)
            if matches:
                scores[intent] = len(matches)
        if not scores:
            return "general"
        return max(scores, key=scores.get)

    # ------------------------------------------------------------------
    # Intent handlers
    # ------------------------------------------------------------------

    def _handle_apaar(self, query: str, month: Optional[str], intent: str) -> Dict[str, Any]:
        trend = self.rag.get_apaar_trend()
        if not trend:
            return self._handle_general(query, month, intent)
        key_points = [f"{t['date']}: {t['total']:,} APAAR IDs generated" for t in trend]
        latest = trend[-1]
        summary = f"APAAR ID Generation: {latest['total']:,} IDs generated as of {latest['date']}."
        return self._build_response(
            intent=intent, summary=summary, key_points=key_points,
            sources=[{"month": t["month"], "section": "apaar", "type": "ground_truth"} for t in trend],
            visualization={"type": "line_chart", "label": "APAAR IDs Generated", "data": trend},
            confidence="high",
        )

    def _handle_dashboard(self, query: str, month: Optional[str], intent: str) -> Dict[str, Any]:
        results = self.rag.search(query, top_k=5)
        dashboard_results = [r for r in results if "dashboard" in r["text"].lower()]
        if not dashboard_results:
            dashboard_results = results[:3]
        key_points = [r["text"] for r in dashboard_results]
        sources = [{"month": r["month"], "section": r["section"], "relevance": r["relevance"],
                     "type": "rag_search"} for r in dashboard_results]
        summary = f"Found {len(dashboard_results)} dashboard-related updates."
        return self._build_response(
            intent=intent, summary=summary, key_points=key_points,
            sources=sources, confidence="high" if dashboard_results else "low",
        )

    def _handle_vsk(self, query: str, month: Optional[str], intent: str) -> Dict[str, Any]:
        results = self.rag.search(query, top_k=7)
        vsk_results = [r for r in results if any(kw in r["text"].lower() for kw in ["vsk", "vidya samiksha", "rvsk"])]
        if not vsk_results:
            vsk_results = results[:3]
        key_points = [r["text"] for r in vsk_results]
        sources = [{"month": r["month"], "section": r["section"], "relevance": r["relevance"],
                     "type": "rag_search"} for r in vsk_results]
        summary = f"Found {len(vsk_results)} VSK/RVSK-related updates."
        return self._build_response(
            intent=intent, summary=summary, key_points=key_points,
            sources=sources, confidence="high" if vsk_results else "low",
        )

    def _handle_state_query(self, query: str, month: Optional[str], intent: str) -> Dict[str, Any]:
        results = self.rag.search(query, top_k=7)
        if not results:
            return self._error_response("No data found matching your query.", intent)
        key_points = [r["text"] for r in results]
        sources = [{"month": r["month"], "section": r["section"], "relevance": r["relevance"],
                     "type": "rag_search"} for r in results]
        summary = f"Found {len(results)} entries related to your query."
        return self._build_response(
            intent=intent, summary=summary, key_points=key_points,
            sources=sources, confidence="high",
        )

    def _handle_policy(self, query: str, month: Optional[str], intent: str) -> Dict[str, Any]:
        results = self.rag.search(query, top_k=5)
        if not results:
            return self._error_response("No policy data found matching your query.", intent)
        key_points = [r["text"] for r in results]
        sources = [{"month": r["month"], "section": r["section"], "relevance": r["relevance"],
                     "type": "rag_search"} for r in results]
        summary = f"Found {len(results)} relevant policy/programme entries."
        return self._build_response(
            intent=intent, summary=summary, key_points=key_points,
            sources=sources, confidence="high" if results else "low",
        )

    def _handle_summary(self, query: str, month: Optional[str], intent: str) -> Dict[str, Any]:
        # Try to detect a specific month in the query
        target = self._extract_month(query) or month
        if not target:
            all_months = self.rag.get_all_months()
            target = all_months[-1] if all_months else None
        if not target:
            return self._error_response("No data available.", intent)
        mdata = self.rag.get_month_data(target)
        if not mdata:
            return self._error_response(f"No data for {target}.", intent)
        key_points = mdata.get("highlights", [])
        summary = f"Newsletter for {target}: {mdata.get('theme', 'N/A')}."
        apaar = mdata.get("apaar_ids_total")
        if apaar:
            key_points.append(f"APAAR IDs: {apaar:,} as of {mdata.get('apaar_ids_date', target)}")
        return self._build_response(
            intent=intent, summary=summary, key_points=key_points,
            sources=[{"month": target, "section": "highlights", "type": "ground_truth"}],
            visualization={"type": "summary_card", "month": target, "theme": mdata.get("theme")},
            confidence="high",
        )

    def _handle_meeting(self, query: str, month: Optional[str], intent: str) -> Dict[str, Any]:
        results = self.rag.search(query, top_k=5)
        meeting_results = [r for r in results if any(kw in r["text"].lower() for kw in ["meeting", "workshop", "session", "visit", "conclave"])]
        if not meeting_results:
            meeting_results = results[:3]
        key_points = [r["text"] for r in meeting_results]
        sources = [{"month": r["month"], "section": r["section"], "relevance": r["relevance"],
                     "type": "rag_search"} for r in meeting_results]
        summary = f"Found {len(meeting_results)} meeting/session-related updates."
        return self._build_response(
            intent=intent, summary=summary, key_points=key_points,
            sources=sources, confidence="high" if meeting_results else "low",
        )

    def _handle_correspondence(self, query: str, month: Optional[str], intent: str) -> Dict[str, Any]:
        results = self.rag.search(query, top_k=5)
        letter_results = [r for r in results if any(kw in r["text"].lower() for kw in ["letter", "request", "reminder", "communication", "proposal"])]
        if not letter_results:
            letter_results = results[:3]
        key_points = [r["text"] for r in letter_results]
        sources = [{"month": r["month"], "section": r["section"], "relevance": r["relevance"],
                     "type": "rag_search"} for r in letter_results]
        summary = f"Found {len(letter_results)} correspondence entries."
        return self._build_response(
            intent=intent, summary=summary, key_points=key_points,
            sources=sources, confidence="high" if letter_results else "low",
        )

    def _handle_help(self, query: str, month: Optional[str], intent: str) -> Dict[str, Any]:
        return self._build_response(
            intent="help",
            summary="I can help you with RVSK newsletter data from the Ministry of Education (April 2025 to January 2026).",
            key_points=[
                "Ask for a monthly summary (e.g., 'Summary for January 2026')",
                "Query APAAR ID generation progress (e.g., 'APAAR IDs trend')",
                "Ask about VSK operations (e.g., 'Bihar VSK status')",
                "Search dashboard developments (e.g., 'NIPUN dashboard')",
                "Find correspondence and letters (e.g., 'letters to states')",
                "Query meetings and workshops (e.g., 'Capacity Building Workshop')",
                "Search by state (e.g., 'Kerala', 'Meghalaya', 'West Bengal')",
            ],
            sources=[],
            confidence="high",
        )

    def _handle_general(self, query: str, month: Optional[str], intent: str) -> Dict[str, Any]:
        results = self.rag.search(query, top_k=5)
        if not results:
            return self._build_response(
                intent=intent,
                summary="I could not find specific data matching your query. Please try rephrasing or ask about APAAR IDs, VSK operations, dashboards, or monthly summaries.",
                key_points=[],
                sources=[],
                confidence="low",
            )
        key_points = [r["text"] for r in results]
        sources = [{"month": r["month"], "section": r["section"],
                     "relevance": r["relevance"], "type": "rag_search"} for r in results]
        summary = f"Found {len(results)} relevant entries from RVSK newsletter data."
        return self._build_response(
            intent=intent, summary=summary, key_points=key_points,
            sources=sources, confidence="medium" if results else "low",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_month(self, query: str) -> Optional[str]:
        """Try to extract a month reference from the query."""
        month_names = ["january", "february", "march", "april", "may", "june",
                       "july", "august", "september", "october", "november", "december"]
        q_lower = query.lower()
        for mname in month_names:
            match = re.search(rf"\b{mname}\s+(\d{{4}})\b", q_lower)
            if match:
                return f"{mname.title()} {match.group(1)}"
            if mname in q_lower:
                # Default to available months
                for m in self.rag.get_all_months():
                    if mname in m.lower():
                        return m
        return None

    def _build_response(self, intent: str, summary: str, key_points: List[str],
                        sources: List[Dict], confidence: str = "medium",
                        visualization: Optional[Dict] = None) -> Dict[str, Any]:
        return {
            "intent": intent,
            "summary": summary,
            "key_points": key_points,
            "sources": sources,
            "visualization": visualization,
            "confidence": confidence,
            "data_backed": confidence in ("high", "medium"),
        }

    def _error_response(self, message: str, intent: str) -> Dict[str, Any]:
        return self._build_response(
            intent=intent, summary=message, key_points=[],
            sources=[], confidence="low",
        )
