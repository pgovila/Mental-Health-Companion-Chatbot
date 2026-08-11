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
import os
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
        "anxious", "anxiety", "worried", "worry", "worrying", "nervous",
        "panic", "panicking", "panicked", "scared",
        "fear", "afraid", "dread", "overthinking", "overwhelmed", "uneasy",
        "tense", "restless", "apprehensive", "phobia", "jittery", "on edge",
        "heart racing", "can't breathe", "suffocating", "racing thoughts",
    ],
    "sadness": [
        "sad", "cry", "crying", "cried", "tears", "unhappy", "depressed", "depression",
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
        "not part of", "not included", "not invited", "everyone but me",
        "moved on without me", "no longer part of", "on my own",
        "don't fit in", "dont fit in", "outside the group",
    ],
    "stress": [
        "stressed", "stress", "pressure", "deadline", "deadlines", "exam", "exams", "test",
        "assignment", "homework", "project", "overloaded", "too much", "burnout",
        "burnt out", "exhausted", "exhausting", "tired", "fatigue", "no sleep", "sleepless",
        "can't sleep", "not sleeping", "overwhelmed", "hectic", "swamped",
        "behind", "failing", "fail", "academic", "grades", "gpa",
        "struggling", "struggle",
    ],
    "anger": [
        "angry", "anger", "furious", "rage", "mad", "frustrated", "frustration",
        "irritated", "irritation", "annoyed", "annoying", "hate", "hatred",
        "unfair", "injustice", "betrayed", "resentful", "bitter", "fuming",
        "livid", "outraged",
    ],
    "joy": [
        "happy", "happiness", "joy", "joyful", "excited", "great", "wonderful",
        "amazing", "fantastic", "awesome", "grateful",
        "thankful", "blessed", "love", "proud", "accomplished", "hopeful",
        "optimistic", "positive", "confident", "energetic", "motivated",
        "inspired", "peaceful", "calm", "relaxed", "content",
    ],
}

# Categories that represent negative-valence emotional states. Used to decide
# whether a message's overall distress signal is strong enough to resist
# being overridden by a possibly-misleading positive polarity score (see
# _classify_emotion below).
_NEGATIVE_VALENCE_EMOTIONS: tuple[Emotion, ...] = ("anxiety", "sadness", "loneliness", "stress", "anger")

# Words that describe a TOPIC (exams, deadlines, work) rather than a FEELING.
# Mentioning one of these alone doesn't tell you the emotional valence — "my
# exam went great" and "my exam is stressing me out" both mention "exam".
# Only words in this set are eligible for the positive-polarity override
# below. Genuine feeling/action words (e.g. "stressed", "panicking",
# "struggling", "exhausted") are NEVER overridden by polarity alone, no
# matter how positive TextBlob scores the sentence — those are direct
# self-reports of distress and take priority over a possibly-misleading
# polarity score (a known TextBlob quirk with intensifier words like "huge"
# or "really").
_WEAK_TOPIC_WORDS: set[str] = {
    "exam", "exams", "test", "assignment", "homework", "project",
    "academic", "grades", "gpa", "deadline", "deadlines",
}

CRISIS_PHRASES: list[str] = [
    "want to die", "kill myself", "end my life", "suicide", "suicidal",
    "self harm", "self-harm", "hurt myself", "no reason to live",
    "can't go on", "give up on life", "don't want to exist",
    "wish i was dead", "better off dead", "end it all",
    # Indirect / passive expressions of suicidal ideation — these are just as
    # serious as explicit phrases and are commonly how distress is first
    # expressed, so they must be caught even without an explicit "die"/"suicide" word.
    "better off without me", "better without me", "world would be better without me",
    "everyone would be better off without me", "no point in living",
    "no point going on", "tired of living", "tired of being alive",
    "i'm a burden", "im a burden", "everyone's better off",
    "nobody would miss me", "no one would miss me", "not meant to be here",
    "shouldn't be here", "should not be here", "disappear forever",
    "wish i wasn't born", "wish i was never born", "life isn't worth it",
    "life isn't worth living", "not worth living",
]

# Word-sequence patterns for the phrases above: each inner list is a phrase
# broken into its core words. A phrase counts as matched if those words all
# appear IN ORDER within a small window (a few words apart), which catches
# real-world variations like "nobody would EVEN miss me" or "I FEEL LIKE a
# burden" that a rigid exact-substring match on CRISIS_PHRASES would miss.
# This is deliberately biased toward over-triggering rather than under-
# triggering: a supportive crisis message shown when not strictly needed is a
# minor inconvenience, but missing real ideation is the failure mode this
# project cannot tolerate.
_CRISIS_WORD_SEQUENCES: list[list[str]] = [
    ["better", "off", "without", "me"],
    ["better", "without", "me"],
    ["nobody", "miss", "me"],
    ["no", "one", "miss", "me"],
    ["burden", "everyone"],
    ["burden", "family"],
    ["burden", "friends"],
    ["burden", "everybody"],
    ["tired", "of", "living"],
    ["tired", "of", "being", "alive"],
    ["no", "point", "living"],
    ["no", "point", "going", "on"],
    ["wish", "i", "was", "never", "born"],
    ["wish", "i", "wasn't", "born"],
    ["everyone", "better", "without", "me"],
]

_MAX_WORD_GAP = 3  # allowed filler words between each matched word


def _build_sequence_pattern(words: list[str]) -> re.Pattern:
    gap = r"(?:\s+\w+){0," + str(_MAX_WORD_GAP) + r"}\s+"
    return re.compile(gap.join(re.escape(w) for w in words))


_CRISIS_SEQUENCE_PATTERNS = [_build_sequence_pattern(seq) for seq in _CRISIS_WORD_SEQUENCES]

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
        self._remote_client = None
        self._remote_client_load_attempted = False

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

    def _get_remote_client(self):
        """
        Lazily creates a HuggingFace InferenceClient that calls the model
        over HTTP instead of loading it locally. This is what lets the free
        Render deployment use real transformer-based classification without
        the RAM cost of torch — the `huggingface_hub` package is tiny (no
        torch dependency) since all it does is send/receive small HTTP
        requests to HuggingFace's own hosted servers.

        Only activates if an HF_API_TOKEN environment variable is set (a
        free HuggingFace account + access token, created at
        https://huggingface.co/settings/tokens). Without it, this silently
        does nothing and the analyzer falls through to keyword-only mode,
        exactly as before.
        """
        if self._remote_client_load_attempted:
            return self._remote_client

        self._remote_client_load_attempted = True
        token = os.environ.get("HF_API_TOKEN")
        if not token:
            self._remote_client = None
            return None

        try:
            from huggingface_hub import InferenceClient
            self._remote_client = InferenceClient(token=token)
        except Exception:
            self._remote_client = None

        return self._remote_client

    def _get_classification(self, text: str) -> list[dict] | None:
        """
        Returns a list of {"label": ..., "score": ...} dicts from whichever
        classification backend is available, trying in order:
          1. Local transformer pipeline (if transformers/torch installed —
             typically only true in local dev via requirements-local.txt)
          2. Remote HuggingFace Inference API (if HF_API_TOKEN is set —
             this is what the free Render deployment uses)
          3. None (caller falls back to pure keyword/polarity classification)
        """
        nlp = self._get_transformer()
        if nlp is not None:
            try:
                return nlp(text)[0]
            except Exception:
                pass  # fall through to remote API attempt below

        client = self._get_remote_client()
        if client is not None:
            try:
                results = client.text_classification(text, model=_MODEL_NAME)
                return [{"label": r.label, "score": r.score} for r in results]
            except Exception:
                pass  # fall through to None -> keyword fallback

        return None

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
        lexicon_emotion, lexicon_keywords, lexicon_score, all_scores, all_matched = self._lexicon_scan(text)

        # Stress and loneliness are NOT in the transformer's label set, so if
        # the lexicon has a solid hit on either, trust it directly.
        if lexicon_emotion in ("stress", "loneliness") and lexicon_score >= 2:
            return lexicon_emotion, lexicon_keywords, min(0.5 + lexicon_score * 0.1, 0.95)

        raw = self._get_classification(text)  # tries local transformer, then remote HF API
        if raw is not None:
            try:
                top = max(raw, key=lambda r: r["score"])
                mapped = self._map_transformer_label(top["label"], polarity)

                # If the transformer landed on "neutral"/"joy" but the lexicon
                # has a strong, specific stress/loneliness signal, prefer the
                # lexicon — the transformer simply can't express those labels.
                # BUT: only trust the lexicon over the transformer when there's
                # real evidence — either 2+ matched words, or at least one
                # genuine feeling word (not just an ambiguous topic word like
                # "exams" or "deadline", which say nothing about valence on
                # their own). Otherwise a single weak topic-word match would
                # override even a confident, correct transformer read — e.g.
                # "my exams went better than expected" (transformer: joy)
                # getting flipped back to "stress" purely because it mentions
                # exams.
                has_real_feeling_word = any(kw not in _WEAK_TOPIC_WORDS for kw in lexicon_keywords)
                if (
                    mapped in ("neutral", "joy")
                    and lexicon_emotion in ("stress", "loneliness")
                    and (lexicon_score >= 2 or has_real_feeling_word)
                ):
                    return lexicon_emotion, lexicon_keywords, 0.55

                keywords = lexicon_keywords if lexicon_emotion == mapped else []
                return mapped, keywords, float(top["score"])
            except Exception:
                pass  # fall through to pure keyword logic below

        # ---- Fallback: original pure keyword + polarity approach ----
        # With ZERO keyword evidence, polarity is the only signal available —
        # but TextBlob's polarity is a weak, easily-misled proxy for genuine
        # emotional content (see the "friends busy with their own lives"
        # case: polarity read positive purely from neutral/positive-adjacent
        # words like "friends", despite the sentence describing exclusion).
        # Requiring a HIGHER polarity magnitude here (0.3 instead of 0.05)
        # before committing to joy/sadness reduces false-confidence labeling
        # on subtle or indirect messages — anything weaker defaults to
        # "neutral" rather than guessing wrong with unwarranted confidence.
        if lexicon_score == 0:
            if polarity > 0.3:
                return "joy", [], 0.35
            if polarity < -0.3:
                return "sadness", [], 0.35
            return "neutral", [], 0.2

        # A weak match shouldn't override a message whose overall sentiment
        # reads as clearly positive — that produces false positives like "my
        # exams went better than expected" being labeled stress just because
        # it mentions exams. We only allow this override when EVERY matched
        # negative-category word across the whole message is a "topic" word
        # (see _WEAK_TOPIC_WORDS) rather than a genuine feeling/action word.
        # If even one real feeling word is present (e.g. "stressed",
        # "panicking", "struggling", "exhausted"), that's treated as a direct
        # self-report of distress and is NEVER overridden by polarity alone —
        # TextBlob's polarity score is known to be misled by intensifier
        # words like "huge" or "really", so it isn't trustworthy enough to
        # overrule an explicit feeling word.
        if (
            all_matched
            and all(kw in _WEAK_TOPIC_WORDS for kw in all_matched)
            and polarity > 0.15
            and lexicon_emotion in _NEGATIVE_VALENCE_EMOTIONS
        ):
            return "joy", [], 0.4

        return lexicon_emotion, lexicon_keywords, min(0.4 + lexicon_score * 0.1, 0.9)

    @staticmethod
    def _map_transformer_label(label: str, polarity: float) -> Emotion:
        label = label.lower()
        if label == "surprise":
            return "joy" if polarity >= 0 else "anxiety"
        return _TRANSFORMER_TO_DOMAIN.get(label, "neutral")

    @staticmethod
    def _lexicon_scan(text: str) -> tuple[Emotion, list[str], int, dict[Emotion, int], list[str]]:
        """Score each emotion by counting matching keywords.

        Returns (best_emotion, best_emotion_keywords, best_emotion_score,
        all_category_scores, all_matched_keywords) — the last two let callers
        see the FULL distress picture across every category, not just
        whichever one "won".
        """
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
        all_matched = [kw for e in _NEGATIVE_VALENCE_EMOTIONS for kw in matched[e]]
        return best_emotion, matched[best_emotion], scores[best_emotion], scores, all_matched

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
        if any(phrase in text for phrase in CRISIS_PHRASES):
            return True
        return any(pattern.search(text) for pattern in _CRISIS_SEQUENCE_PATTERNS)