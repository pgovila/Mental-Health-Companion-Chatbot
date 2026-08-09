"""
mood_tracker.py
---------------
Tracks mood data across a session and produces a summary report.
"""

from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter
from .sentiment_analyzer import AnalysisResult, Emotion, Sentiment


@dataclass
class MoodEntry:
    timestamp:  datetime
    sentiment:  Sentiment
    emotion:    Emotion
    polarity:   float
    intensity:  float
    snippet:    str         # first 60 chars of user message


class MoodTracker:
    """
    Records each turn's analysis result and summarises the session.
    """

    def __init__(self) -> None:
        self._entries: list[MoodEntry] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, result: AnalysisResult) -> None:
        self._entries.append(MoodEntry(
            timestamp = datetime.now(),
            sentiment = result.sentiment,
            emotion   = result.emotion,
            polarity  = result.polarity,
            intensity = result.intensity,
            snippet   = result.raw_text[:60],
        ))

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def current_sentiment(self) -> Sentiment | None:
        if self._entries:
            return self._entries[-1].sentiment
        return None

    def average_polarity(self) -> float:
        if not self._entries:
            return 0.0
        return round(sum(e.polarity for e in self._entries) / len(self._entries), 3)

    def dominant_emotion(self) -> Emotion:
        if not self._entries:
            return "neutral"
        counts = Counter(e.emotion for e in self._entries)
        return counts.most_common(1)[0][0]

    def sentiment_distribution(self) -> dict[Sentiment, int]:
        counts: dict[Sentiment, int] = {"positive": 0, "negative": 0, "neutral": 0}
        for e in self._entries:
            counts[e.sentiment] += 1
        return counts

    def trend(self) -> str:
        """
        Returns 'improving', 'declining', or 'stable' based on polarity
        slope across the session.
        """
        if len(self._entries) < 3:
            return "stable"

        polarities = [e.polarity for e in self._entries]
        first_half = sum(polarities[: len(polarities) // 2]) / (len(polarities) // 2)
        second_half = sum(polarities[len(polarities) // 2 :]) / (len(polarities) - len(polarities) // 2)

        delta = second_half - first_half
        if delta > 0.1:
            return "improving"
        if delta < -0.1:
            return "declining"
        return "stable"

    def polarity_history(self) -> list[float]:
        return [e.polarity for e in self._entries]

    def summary_text(self) -> str:
        """Returns a formatted plain-text mood summary for the session."""
        if not self._entries:
            return "No mood data recorded this session."

        dist   = self.sentiment_distribution()
        total  = self.entry_count
        trend  = self.trend()
        avg_p  = self.average_polarity()
        dom_e  = self.dominant_emotion()

        trend_emoji = {"improving": "📈", "declining": "📉", "stable": "➡️"}[trend]

        lines = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "        📊 SESSION MOOD SUMMARY",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  Messages analysed : {total}",
            f"  Dominant emotion  : {dom_e.upper()}",
            f"  Average polarity  : {avg_p:+.3f}",
            f"  Mood trend        : {trend.capitalize()} {trend_emoji}",
            "",
            "  Sentiment breakdown:",
            f"    😊 Positive : {dist['positive']} ({100*dist['positive']//total if total else 0}%)",
            f"    😟 Negative : {dist['negative']} ({100*dist['negative']//total if total else 0}%)",
            f"    😐 Neutral  : {dist['neutral']} ({100*dist['neutral']//total if total else 0}%)",
        ]

        # Closing note based on dominant emotion
        if avg_p > 0.15:
            lines += ["", "  You had a positive session! Keep it up! 🌟"]
        elif avg_p < -0.15:
            lines += ["", "  It sounds like today was tough. Please be kind to yourself. 💙"]
        else:
            lines += ["", "  A balanced session. Remember to check in with yourself regularly. 🌿"]

        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)

    def ascii_polarity_chart(self) -> str:
        """
        Renders a simple ASCII line chart of polarity over the session.
        Only shown when there are ≥ 3 entries.
        """
        history = self.polarity_history()
        if len(history) < 3:
            return ""

        HEIGHT = 7      # rows
        WIDTH  = min(len(history), 40)

        # Map polarity (-1 → 1) to row (0 → HEIGHT-1), inverted
        def to_row(p: float) -> int:
            normalized = (p + 1.0) / 2.0        # 0 → 1
            return int((1.0 - normalized) * (HEIGHT - 1))

        # Sample to WIDTH if there are more entries
        step = max(1, len(history) // WIDTH)
        sampled = history[::step][:WIDTH]

        grid = [[" "] * len(sampled) for _ in range(HEIGHT)]
        for col, p in enumerate(sampled):
            row = to_row(p)
            grid[row][col] = "●"

        mid = HEIGHT // 2
        labels = [f" {'+1.0':>4}" if r == 0 else (f" {'0.0':>4}" if r == mid else (f" {'-1.0':>4}" if r == HEIGHT - 1 else "      ")) for r in range(HEIGHT)]

        lines = ["  Polarity over time:"]
        for r in range(HEIGHT):
            lines.append(f"{labels[r]} │{''.join(grid[r])}")
        lines.append("       └" + "─" * len(sampled))
        lines.append(f"        1{'':>{len(sampled)-2}}{len(sampled)}")
        return "\n".join(lines)
