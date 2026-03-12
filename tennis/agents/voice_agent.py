"""
Voice Agent — generates a timed narration script for the AI tennis coach.

Produces coaching commentary tied to specific timestamps in the video.
These events are played back through the browser's Web Speech API or
an optional TTS service (OpenAI TTS if OPENAI_API_KEY is set).
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


# Narration templates per event type
NARRATION_TEMPLATES = {
    "intro": [
        "Let me walk you through your match analysis.",
        "Here's your personalized coaching breakdown.",
        "I've analyzed your match. Let's look at the key moments.",
    ],
    "strength": [
        "Nice shot here — your {shot} technique is looking solid.",
        "Good use of the {shot}. Keep that up.",
        "That's a confident {shot}. Well done.",
    ],
    "issue_late_contact": [
        "Notice how the contact point is slightly late here. Try to meet the ball earlier.",
        "You're catching the ball a bit late. Step in sooner and contact the ball out front.",
    ],
    "issue_weak_followthrough": [
        "The follow-through stops short here. Drive the racket all the way over your shoulder.",
        "Complete that swing — a full follow-through generates topspin and consistency.",
    ],
    "issue_poor_footwork": [
        "Watch your feet here — you're a little flat-footed. Use a split step when your opponent contacts.",
        "Better footwork here would set you up for a stronger shot. Smaller, quicker adjustment steps.",
    ],
    "issue_high_backswing": [
        "The backswing is a bit high and loopy. A more compact backswing gives you better control.",
        "Keep the backswing lower — this will give you more time to adjust to fast balls.",
    ],
    "issue_poor_positioning": [
        "Court position here could improve — recover toward the center after each shot.",
        "You're caught wide. After this shot, push off toward the center mark to recover.",
    ],
    "rally_moment": [
        "Good rallying here — you're maintaining consistent depth.",
        "This rally shows your baseline consistency. Keep that ball deep.",
    ],
    "summary": [
        "Overall, a solid session. Focus on {top_issue} as your primary improvement target.",
        "Great effort today. Work on {top_issue} and you'll see real results quickly.",
        "That's the full analysis. Your main focus should be on {top_issue}.",
    ],
}


class VoiceAgent:
    """Expert agent that generates the timed voice coaching narration script."""

    name = "voice"

    def analyze(self, frames: list, context: dict) -> dict:
        """
        Build a timed narration script from all coaching insights.

        Returns:
            script: list of {timestamp, text, event_type}
            intro_text: str
            summary_text: str
            total_cues: int
        """
        bio = context.get("biomechanics", {})
        coaching = context.get("coaching", {})
        ball = context.get("ball", {})
        duration = context.get("duration_seconds", 30.0)

        frame_annotations = bio.get("frame_annotations", [])
        issues = bio.get("issues", [])
        recommendations = coaching.get("recommendations", [])
        overall_rating = coaching.get("overall_rating", "Intermediate")
        overall_score = coaching.get("overall_score", 6.5)
        shot_scores = bio.get("shot_quality_scores", {})

        script = []

        # Intro narration at the start
        intro = (
            f"Welcome to your TennisIQ coaching review. "
            f"I've analyzed your {context.get('match_type', 'session')} and rated you at "
            f"{overall_rating} level with a score of {overall_score:.1f} out of 10. "
            f"Let's go through the key moments."
        )
        script.append({"timestamp": 0.5, "text": intro, "event_type": "intro"})

        # Narration for each frame annotation (technique issues)
        for ann in frame_annotations:
            issue_id = ann.get("issue_id", "")
            template_key = f"issue_{issue_id}" if f"issue_{issue_id}" in NARRATION_TEMPLATES else "rally_moment"
            templates = NARRATION_TEMPLATES.get(template_key, NARRATION_TEMPLATES["rally_moment"])
            text = templates[len(script) % len(templates)]

            script.append({
                "timestamp": ann["timestamp"],
                "text": text,
                "event_type": "coaching",
                "issue_id": issue_id,
                "severity": ann.get("severity", "medium"),
            })

        # Halfway progress narration
        if duration > 15:
            best_shot = max(shot_scores, key=shot_scores.get) if shot_scores else "forehand"
            mid_text = NARRATION_TEMPLATES["strength"][1 % len(NARRATION_TEMPLATES["strength"])]
            script.append({
                "timestamp": round(duration * 0.5, 1),
                "text": mid_text.format(shot=best_shot),
                "event_type": "strength",
            })

        # Summary narration at the end
        top_issue = issues[0]["label"] if issues else "technique consistency"
        summary_template = NARRATION_TEMPLATES["summary"][len(issues) % 3]
        summary = summary_template.format(top_issue=top_issue)
        full_summary = (
            f"{summary} "
            f"Your top recommendation: {recommendations[0]['detail'] if recommendations else 'Continue practicing consistently'}."
        )
        script.append({
            "timestamp": max(duration - 3.0, duration * 0.9),
            "text": full_summary,
            "event_type": "summary",
        })

        # Sort chronologically
        script.sort(key=lambda e: e["timestamp"])

        return {
            "script": script,
            "intro_text": intro,
            "summary_text": full_summary,
            "total_cues": len(script),
        }
