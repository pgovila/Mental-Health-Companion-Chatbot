"""
chatbot.py
----------
Core chatbot orchestration — wires together the analyzer, response engine,
relaxation tips, mood tracker, and crisis handler.
"""

from .sentiment_analyzer import SentimentAnalyzer, AnalysisResult
from .response_engine     import ResponseEngine
from .relaxation_tips     import RelaxationTips
from .mood_tracker        import MoodTracker
from .crisis_handler      import CrisisHandler


# ---------------------------------------------------------------------------
# Chatbot Core
# ---------------------------------------------------------------------------

class MentalHealthChatbot:
    """
    Orchestrates a single conversation session.

    Usage:
        bot = MentalHealthChatbot()
        output = bot.respond("I'm feeling really anxious about my exams")
        print(output.text)
    """

    def __init__(self) -> None:
        self._analyzer  = SentimentAnalyzer()
        self._engine    = ResponseEngine()
        self._tips      = RelaxationTips()
        self._tracker   = MoodTracker()
        self._crisis    = CrisisHandler()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def respond(self, user_input: str) -> "ChatResponse":
        """
        Process a user message and return a ChatResponse.
        """
        text = user_input.strip()

        # ── Special commands ────────────────────────────────────────────
        lower = text.lower()
        if lower in ("quit", "exit", "bye", "goodbye"):
            return ChatResponse(
                text          = self._farewell(),
                analysis      = None,
                show_tip      = False,
                is_farewell   = True,
                mood_summary  = self._tracker.summary_text(),
                ascii_chart   = self._tracker.ascii_polarity_chart(),
            )

        if lower == "mood":
            return ChatResponse(
                text         = self._tracker.summary_text() + "\n" + self._tracker.ascii_polarity_chart(),
                analysis     = None,
                show_tip     = False,
                is_farewell  = False,
            )

        if lower == "relax":
            return ChatResponse(
                text         = self._tips.get_random_tip(),
                analysis     = None,
                show_tip     = False,
                is_farewell  = False,
            )

        if lower == "quote":
            return ChatResponse(
                text         = self._engine.get_motivational_quote(),
                analysis     = None,
                show_tip     = False,
                is_farewell  = False,
            )

        if lower == "help":
            return ChatResponse(
                text         = HELP_TEXT,
                analysis     = None,
                show_tip     = False,
                is_farewell  = False,
            )

        # ── Normal conversational turn ───────────────────────────────────
        if not text:
            return ChatResponse(
                text         = "I'm here — feel free to share anything on your mind. 💙",
                analysis     = None,
                show_tip     = False,
                is_farewell  = False,
            )

        result = self._analyzer.analyze(text)
        self._tracker.record(result)

        # Crisis path
        if result.is_crisis:
            return ChatResponse(
                text         = self._crisis.get_response(),
                analysis     = result,
                show_tip     = False,
                is_crisis    = True,
                is_farewell  = False,
            )

        # Normal empathetic response
        response_text = self._engine.generate(result)

        # Decide whether to append a relaxation tip
        should_offer_tip = (
            result.emotion in {"anxiety", "stress", "anger"}
            and result.intensity >= 0.35
        )

        tip_text = None
        if should_offer_tip:
            tip_text = self._tips.get_tip(result.emotion)

        return ChatResponse(
            text         = response_text,
            analysis     = result,
            show_tip     = should_offer_tip,
            tip_text     = tip_text,
            is_farewell  = False,
        )

    def session_summary(self) -> str:
        return self._tracker.summary_text()

    def ascii_chart(self) -> str:
        return self._tracker.ascii_polarity_chart()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _farewell(self) -> str:
        sentiment = self._tracker.current_sentiment()
        if sentiment == "positive":
            return (
                "It was wonderful chatting with you today! 🌟 "
                "You're leaving on a high note — keep that energy going! "
                "Take care and come back whenever you need. 💙"
            )
        if sentiment == "negative":
            return (
                "Thank you for trusting me today. I hope our conversation brought "
                "even a small moment of comfort. Please take care of yourself, reach out "
                "to someone you trust, and remember: you are not alone. 💙"
            )
        return (
            "Thank you for spending time with me today. "
            "Remember to take care of your mental health — it matters. "
            "You're always welcome back. Take care! 🌿"
        )


# ---------------------------------------------------------------------------
# Response Data Class
# ---------------------------------------------------------------------------

class ChatResponse:
    __slots__ = (
        "text", "analysis", "show_tip", "tip_text",
        "is_farewell", "is_crisis", "mood_summary", "ascii_chart",
    )

    def __init__(
        self,
        text:         str,
        analysis:     AnalysisResult | None,
        show_tip:     bool,
        is_farewell:  bool,
        tip_text:     str | None       = None,
        is_crisis:    bool             = False,
        mood_summary: str | None       = None,
        ascii_chart:  str | None       = None,
    ) -> None:
        self.text         = text
        self.analysis     = analysis
        self.show_tip     = show_tip
        self.tip_text     = tip_text
        self.is_farewell  = is_farewell
        self.is_crisis    = is_crisis
        self.mood_summary = mood_summary
        self.ascii_chart  = ascii_chart


# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------

HELP_TEXT = """\
┌─────────────────────────────────────────────┐
│         MENTAL HEALTH COMPANION — HELP       │
├─────────────────────────────────────────────┤
│  Just type how you're feeling and I'll       │
│  respond with empathy and support.           │
│                                              │
│  Special commands:                           │
│   mood    → View your session mood chart     │
│   relax   → Get a relaxation exercise        │
│   quote   → Get a motivational quote         │
│   help    → Show this help message           │
│   quit    → End session & see mood summary   │
│                                              │
│  💡 Tip: Be honest — the more you share,     │
│     the better I can support you.            │
└─────────────────────────────────────────────┘"""
