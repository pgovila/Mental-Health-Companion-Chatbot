"""
sentiment_analyzer.py
---------------------
Detects sentiment polarity and classifies emotion from user text.

Hybrid pipeline:
  - TextBlob        -> polarity (-1..+1) and subjectivity (0..1). Cheap, fast,
                        good enough for a continuous "how positive/negative" signal.
  - HuggingFace      -> j-hartmann/emotion-english-distilroberta-base gives a
    transformer         calibrated probability distribution over 7 general
                         emotions (anger, disgust, fear, joy, neutral, sadness,
                         surprise). This replaces the old pure keyword-matching
                         approach for the emotions it covers, which is far more
                         robust to phrasing the keyword lexicon never saw.
  - Keyword lexicon  -> the transformer model has no concept of "stress" or
                        "loneliness" (they aren't in its label set), so the
                        lexicon is kept specifically to catch those two
                        domain-specific categories. If the lexicon fires
                        strongly for stress/loneliness and the transformer
                        isn't highly confident in something else, the lexicon
                        wins. Otherwise the transformer's mapped label wins.

This keeps the same AnalysisResult shape as before, so chatbot.py,
mood_tracker.py, response_engine.py, app.py, and main.py all continue to work
unmodified.

If the transformer model can't be loaded (e.g. offline, or a stripped-down
deployment without torch), the analyzer automatically falls back to the
original pure keyword + polarity approach so the app never hard-crashes.
"""

from dataclasses import dataclass, field
from typing import Literal
import re

try:
    from textblob import TextBlob
except ImportError:
    raise ImportError("Please install dependencies: pip install -r requirements.txt")


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Sentiment = Literal["positive", "negative", "neutral"]
Emotion   = Literal["anxiety", "sadness", "loneliness", "stress", "anger", "joy", "neutral"]


@dataclass
class AnalysisResult:
    raw_text:       str
    sentiment:      Sentiment
    polarity:       float          # -1.0 (very negative) → +1.0 (very positive)
    subjectivity:   float          # 0.0 (objective) → 1.0 (subjective)
    emotion:        Emotion
    intensity:      float          # 0.0 → 1.0 — how strongly the emotion is expressed
    keywords:       list[str] = field(default_factory=list)
    is_crisis:      bool = False


# ---------------------------------------------------------------------------
# Emotion keyword lexicon (kept for stress/loneliness detection + fallback)
# ---------------------------------------------------------------------------

EMOTION_LEXICON: dict[Emotion, list[str]] = {
    "anxiety": [
        "anxious", "anxiety", "worried", "worry", "nervous", "panic", "scared",
        "fear", "afraid", "dread", "overthinking", "overwhelmed", "uneasy",
        "tense", "restless", "apprehensive", "phobia", "jittery", "on edge",
        "heart racing", "can't breathe", "suffocating", "racing thoughts",
    ],
    "sadness": [
        "sad", "cry", "crying", "tears", "unhappy", "depressed", "depression",
        "miserable", "heartbroken", "grief", "grieve", "lost", "hopeless",
        "hopelessness", "empty", "numb", "worthless", "broken", "shattered",
        "despair", "despairing", "low", "down", "gloomy", "sorrow", "sorrowful",
        "melancholy", "devastated", "hurt", "pain",
    ],
    "loneliness": [
        "lonely", "alone", "loneliness", "isolated", "isolation", "no friends",
        "nobody cares", "no one", "forgotten", "left out", "excluded",
        "abandoned", "invisible", "disconnected", "solitude", "empty room",
        "miss people", "missing connection", "by myself",
    ],
    "stress": [
        "stressed", "stress", "pressure", "deadline", "exam", "exams", "test",
        "assignment", "homework", "project", "overloaded", "too much", "burnout",
        "burnt out", "exhausted", "tired", "fatigue", "no sleep", "sleepless",
        "can't sleep", "not sleeping", "overwhelmed", "hectic", "swamped",
        "behind", "failing", "fail", "academic", "grades", "gpa",
    ],
    "anger": [
        "angry", "anger", "furious", "rage", "mad", "frustrated", "frustration",
        "irritated", "irritation", "annoyed", "annoying", "hate", "hatred",
        "unfair", "injustice", "betrayed", "resentful", "bitter", "fuming",
        "livid", "outraged",
    ],
    "joy": [
        "happy", "happiness", "joy", "joyful", "excited", "great", "wonderful",
        "amazing", "fantastic", "awesome", "good", "better", "grateful",
        "thankful", "blessed", "love", "proud", "accomplished", "hopeful",
        "optimistic", "positive", "confident", "energetic", "motivated",
        "inspired", "peaceful", "calm", "relaxed", "content",
    ],
}

CRISIS_PHRASES: list[str] = [
    "want to die", "kill myself", "end my life", "suicide", "suicidal",
    "self harm", "self-harm", "hurt myself", "no reason to live",
    "can't go on", "give up on life", "don't want to exist",
    "wish i was dead", "better off dead", "end it all",
]

# Maps the transformer's 7 general emotion labels onto our 7 domain labels.
# "disgust" and "surprise" have no direct equivalent in our taxonomy, so they
# are routed based on polarity at classification time (see _map_transformer_label).
_TRANSFORMER_TO_DOMAIN: dict[str, Emotion] = {
    "fear":     "anxiety",
    "sadness":  "sadness",
    "anger":    "anger",
    "joy":      "joy",
    "neutral":  "neutral",
    "disgust":  "anger",       # closest available match
}

_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """
    Combines TextBlob polarity with a transformer emotion classifier
    (falling back to keyword classification if the model is unavailable)
    to produce a rich AnalysisResult for every user message.
    """

    def __init__(self) -> None:
        self._transformer_pipeline = None
        self._transformer_load_attempted = False

    def analyze(self, text: str) -> AnalysisResult:
        cleaned   = self._clean(text)
        blob      = TextBlob(cleaned)
        polarity  = blob.sentiment.polarity       # -1 → 1
        subj      = blob.sentiment.subjectivity   # 0  → 1

        sentiment = self._polarity_to_sentiment(polarity)
        emotion, keywords, confidence = self._classify_emotion(cleaned, polarity)
        intensity = self._compute_intensity(polarity, subj, keywords, confidence)
        is_crisis = self._check_crisis(cleaned)

        return AnalysisResult(
            raw_text    = text,
            sentiment   = sentiment,
            polarity    = round(polarity, 3),
            subjectivity= round(subj, 3),
            emotion     = emotion,
            intensity   = round(intensity, 3),
            keywords    = keywords,
            is_crisis   = is_crisis,
        )

    # ------------------------------------------------------------------
    # Transformer loading (lazy, so app startup stays fast and doesn't
    # require the model to load unless a message actually gets analyzed)
    # ------------------------------------------------------------------

    def _get_transformer(self):
        if self._transformer_load_attempted:
            return self._transformer_pipeline

        self._transformer_load_attempted = True
        try:
            from transformers import pipeline
            self._transformer_pipeline = pipeline(
                "text-classification",
                model=_MODEL_NAME,
                top_k=None,          # return scores for all labels
                truncation=True,
            )
        except Exception:
            # Covers ImportError (transformers/torch not installed), OSError
            # (no internet / model not cached), and any runtime load failure.
            # We deliberately fail open to keyword-only mode rather than crash.
            self._transformer_pipeline = None

        return self._transformer_pipeline

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(text: str) -> str:
        """Lowercase and strip punctuation clutter (keep apostrophes)."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s']+", " ", text)
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def _polarity_to_sentiment(polarity: float) -> Sentiment:
        if polarity > 0.05:
            return "positive"
        if polarity < -0.05:
            return "negative"
        return "neutral"

    def _classify_emotion(
        self,
        text: str,
        polarity: float,
    ) -> tuple[Emotion, list[str], float]:
        """
        Returns (emotion, matched_keywords, confidence).
        confidence is the transformer's top-label probability if the
        transformer ran, otherwise a keyword-density-derived proxy.
        """
        lexicon_emotion, lexicon_keywords, lexicon_score = self._lexicon_scan(text)

        # Stress and loneliness are NOT in the transformer's label set, so if
        # the lexicon has a solid hit on either, trust it directly.
        if lexicon_emotion in ("stress", "loneliness") and lexicon_score >= 2:
            return lexicon_emotion, lexicon_keywords, min(0.5 + lexicon_score * 0.1, 0.95)

        nlp = self._get_transformer()
        if nlp is not None:
            try:
                raw = nlp(text)[0]  # list[{"label": ..., "score": ...}]
                top = max(raw, key=lambda r: r["score"])
                mapped = self._map_transformer_label(top["label"], polarity)

                # If the transformer landed on "neutral"/"joy" but the lexicon
                # has a strong, specific stress/loneliness signal, prefer the
                # lexicon — the transformer simply can't express those labels.
                if mapped in ("neutral", "joy") and lexicon_emotion in ("stress", "loneliness") and lexicon_score >= 1:
                    return lexicon_emotion, lexicon_keywords, 0.55

                keywords = lexicon_keywords if lexicon_emotion == mapped else []
                return mapped, keywords, float(top["score"])
            except Exception:
                pass  # fall through to pure keyword logic below

        # ---- Fallback: original pure keyword + polarity approach ----
        if lexicon_score == 0:
            if polarity > 0.05:
                return "joy", [], 0.4
            if polarity < -0.05:
                return "sadness", [], 0.4
            return "neutral", [], 0.3
        return lexicon_emotion, lexicon_keywords, min(0.4 + lexicon_score * 0.1, 0.9)

    @staticmethod
    def _map_transformer_label(label: str, polarity: float) -> Emotion:
        label = label.lower()
        if label == "surprise":
            return "joy" if polarity >= 0 else "anxiety"
        return _TRANSFORMER_TO_DOMAIN.get(label, "neutral")

    @staticmethod
    def _lexicon_scan(text: str) -> tuple[Emotion, list[str], int]:
        """Score each emotion by counting matching keywords."""
        scores: dict[Emotion, int] = {e: 0 for e in EMOTION_LEXICON}
        matched: dict[Emotion, list[str]] = {e: [] for e in EMOTION_LEXICON}

        words = set(text.split())
        for emotion, keywords in EMOTION_LEXICON.items():
            for kw in keywords:
                if " " in kw:           # multi-word phrase
                    if kw in text:
                        scores[emotion] += 2
                        matched[emotion].append(kw)
                else:
                    if kw in words:
                        scores[emotion] += 1
                        matched[emotion].append(kw)

        best_emotion: Emotion = max(scores, key=lambda e: scores[e])
        return best_emotion, matched[best_emotion], scores[best_emotion]

    @staticmethod
    def _compute_intensity(
        polarity: float,
        subjectivity: float,
        keywords: list[str],
        confidence: float,
    ) -> float:
        """
        Blends absolute polarity, subjectivity, keyword density, and
        transformer confidence into a 0→1 intensity score.
        """
        kw_density = min(len(keywords) / 5.0, 1.0)
        raw = (
            (abs(polarity) * 0.3)
            + (subjectivity * 0.2)
            + (kw_density * 0.2)
            + (confidence * 0.3)
        )
        return min(raw, 1.0)

    @staticmethod
    def _check_crisis(text: str) -> bool:
        return any(phrase in text for phrase in CRISIS_PHRASES)
