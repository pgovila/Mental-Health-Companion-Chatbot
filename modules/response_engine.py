"""
response_engine.py
------------------
Generates empathetic, contextual responses based on sentiment analysis results.
Uses template pools with polarity-weighted random selection.
"""

import random
from .sentiment_analyzer import AnalysisResult, Emotion


# ---------------------------------------------------------------------------
# Response template pools — keyed by emotion
# ---------------------------------------------------------------------------

RESPONSE_TEMPLATES: dict[str, list[str]] = {

    # ── Anxiety ─────────────────────────────────────────────────────────────
    "anxiety_low": [
        "It sounds like you have a lot on your mind right now. That's completely understandable — worrying sometimes means you care deeply. Take a slow breath and remember: one step at a time. 💙",
        "Feeling a bit uneasy? That's okay. Even small worries deserve acknowledgment. You're doing better than you think — try grounding yourself with the '5-4-3-2-1' technique.",
        "I can sense a little unease in your words. It's alright to feel this way. Try focusing on what's within your control right now — you've got this. 🌿",
    ],
    "anxiety_high": [
        "I hear you — anxiety can feel overwhelming, especially when everything seems to pile up at once. You are not alone in this. Let's slow down together: breathe in for 4 counts, hold for 4, out for 4. Repeat that with me. 💙",
        "That sounds really hard. Anxiety has a way of making everything feel urgent and scary at the same time. Please know: this feeling is temporary and you have the strength to get through it. A short breathing exercise can help right now.",
        "Your feelings are valid and real. When anxiety peaks like this, your nervous system is trying to protect you — but we can gently calm it. Let's try a quick relaxation exercise together. You're safe here. 🤍",
    ],

    # ── Sadness ──────────────────────────────────────────────────────────────
    "sadness_low": [
        "I'm sorry you're feeling a bit low today. It's okay to not be okay sometimes — your feelings are valid. Be gentle with yourself. 🌸",
        "Some days are harder than others, and that's perfectly human. Allow yourself to feel this without judgment. A little kindness toward yourself goes a long way. 💛",
        "Feeling down can be exhausting. Remember, even the darkest clouds pass. You've made it through hard days before, and you will again. 🌤️",
    ],
    "sadness_high": [
        "I'm really glad you're sharing this with me. Feeling this level of sadness is heavy, and I want you to know you don't have to carry it alone. Please reach out to someone you trust — or a counselor — when you're ready. 💙",
        "That sounds deeply painful, and I'm truly sorry you're going through this. Your feelings matter. While I'm here to listen, please consider talking to a counselor or a close friend — you deserve real human support. 🤍",
        "What you're feeling sounds very difficult. Deep sadness can make everything feel hopeless, but that feeling is not the truth — it's a signal that you need care and support. You deserve that support. 💙",
    ],

    # ── Loneliness ───────────────────────────────────────────────────────────
    "loneliness_low": [
        "Feeling a bit disconnected is something many students experience, especially during busy or stressful periods. You're not as alone as you might feel right now. 🤝",
        "Sometimes life gets isolating, especially with the pressures of student life. Know that reaching out — even to a chatbot! — takes courage. I'm here with you. 🌟",
        "Loneliness can sneak up on us. Even small moments of connection — a message to a friend, a smile at a classmate — can shift things. You're more connected than you know. 💛",
    ],
    "loneliness_high": [
        "Feeling deeply alone is one of the most painful human experiences. I want you to know: I see you, and what you feel matters. Please consider reaching out to your university's student support service or a counselor — you deserve genuine connection. 💙",
        "Isolation can make us feel invisible, but you are not invisible to me. You reached out today and that matters. Please try to connect with even one person today — a family member, a classmate, or a helpline. You are worth knowing. 🤍",
        "I hear how alone you feel, and I'm genuinely sorry. That kind of loneliness can weigh so heavily. Please know: many others feel this way in silence. Seeking connection — with support services or peers — is a brave and worthwhile step. 💛",
    ],

    # ── Stress ───────────────────────────────────────────────────────────────
    "stress_low": [
        "Student life can definitely pile on the pressure! You're managing a lot. Breaking things into smaller tasks and taking short breaks can make a real difference. You've got this. 📚",
        "A little stress can actually sharpen focus — so you're channeling it well. Remember to take breathing breaks between tasks and drink water. Small things keep the engine running! 💪",
        "I can sense some academic pressure in your message. That's completely normal. Prioritize, breathe, and remember: you don't have to do everything perfectly — done is often better than perfect. 🌿",
    ],
    "stress_high": [
        "That sounds like an overwhelming amount of pressure. When stress reaches this level, your body and mind are asking for a pause. Please step away from your screen for 5 minutes, take some deep breaths, and remind yourself: your worth is not your grades. 💙",
        "High stress can make your brain feel completely foggy and stuck. You're not weak — your system is overloaded. Let's reset: close your books for a moment, do 5 slow deep breaths, and write down just ONE thing to tackle next. One thing. 💛",
        "I hear how much is on your plate right now. Chronic stress needs more than a quick fix — please talk to a counselor, academic advisor, or even a trusted friend about what's happening. You don't have to hold all of this alone. 🤍",
    ],

    # ── Anger ────────────────────────────────────────────────────────────────
    "anger_low": [
        "Feeling frustrated is a valid and natural response — especially when things feel unfair or out of your control. Acknowledge the feeling, then see if there's one small action you can take. 🌱",
        "A bit of frustration can be a signal that something important needs your attention. What's underneath the anger? Sometimes naming it helps release some pressure. 💛",
        "That sounds annoying and unfair. Your frustration makes sense. When you feel ready, try channeling that energy into something constructive — even a short walk can help shift the feeling. 🌿",
    ],
    "anger_high": [
        "I can feel the intensity of what you're going through. Strong anger often signals deep hurt or injustice. Please give yourself permission to cool down before acting — take a few very slow breaths, then we can work through this. 💙",
        "That sounds really intense. When anger peaks like this, the most helpful thing is to pause before reacting. Find a safe outlet — a brisk walk, writing it out, squeezing a pillow — before making any decisions. 🤍",
        "Anger this strong deserves to be heard. Something has really hurt or pushed you, and that matters. Please be gentle with yourself and others right now, and consider talking to a counselor about what's driving this. 💛",
    ],

    # ── Joy ──────────────────────────────────────────────────────────────────
    "joy": [
        "That's wonderful to hear! 🌟 Your positivity is genuinely uplifting. Savour this feeling — you deserve it!",
        "It's so great that you're feeling good today! 🎉 Keep riding this wave of positivity. You're doing amazing!",
        "Your energy is contagious! 😊 These good moments are worth celebrating. Hold onto this feeling — you've earned it.",
        "Love hearing this! 💛 You're shining today. Keep going — great things are ahead for you.",
    ],

    # ── Neutral ──────────────────────────────────────────────────────────────
    "neutral": [
        "Thank you for sharing that with me. I'm here and listening — feel free to tell me more about what's on your mind. 🌿",
        "I appreciate you talking to me. I'm here for whatever you need — whether that's a chat, some advice, or just someone to listen. 💙",
        "Got it. I'm here for you. Is there anything specific you'd like to explore or talk about today? I'm all ears. 🤍",
        "Thanks for sharing. How are you feeling overall today? I'd love to understand more about what's going on for you. 💛",
    ],

    # ── Crisis ───────────────────────────────────────────────────────────────
    "crisis": [
        "I'm really glad you reached out, and I'm taking what you said seriously. What you're feeling right now is important. Please reach out to a crisis helpline immediately — talking to a real person can make a real difference.\n\n🆘 **iCall (India):** 9152987821\n🆘 **Vandrevala Foundation:** 1860-2662-345\n🆘 **Crisis Text Line (US):** Text HOME to 741741\n🆘 **Samaritans (UK):** 116 123\n\nYou matter. Please reach out. 💙",
        "Thank you for trusting me with this. Please know that you are not alone, and there are people who care and can help right now. Please call or text a crisis line — trained counselors are available 24/7 and want to hear from you.\n\n🆘 **National Suicide Prevention (US):** 988 or 1-800-273-8255\n🆘 **iCall (India):** 9152987821\n🆘 **Crisis Text Line:** Text HOME to 741741\n\nYour life has value. Please reach out now. 🤍",
    ],
}

MOTIVATIONAL_QUOTES: list[str] = [
    "\"You don't have to see the whole staircase, just take the first step.\" — Martin Luther King Jr.",
    "\"In the middle of difficulty lies opportunity.\" — Albert Einstein",
    "\"You are braver than you believe, stronger than you seem, and smarter than you think.\" — A.A. Milne",
    "\"The greatest glory in living lies not in never falling, but in rising every time we fall.\" — Nelson Mandela",
    "\"It does not matter how slowly you go as long as you do not stop.\" — Confucius",
    "\"Believe you can and you're halfway there.\" — Theodore Roosevelt",
    "\"You are enough, a thousand times enough.\" — Atticus",
    "\"Start where you are. Use what you have. Do what you can.\" — Arthur Ashe",
    "\"Be gentle with yourself. You are a child of the universe, no less than the trees and the stars.\" — Max Ehrmann",
    "\"This too shall pass.\" — Persian Proverb",
    "\"You've survived 100% of your worst days so far. You're doing great.\" — Unknown",
    "\"Even the darkest night will end and the sun will rise.\" — Victor Hugo",
    "\"You are not your illness. You have an individual story to tell.\" — Julian Seifter",
    "\"Mental health is not a destination, but a process. It's about how you drive, not where you're going.\" — Noam Shpancer",
    "\"There is hope, even when your brain tells you there isn't.\" — John Green",
]

AFFIRMATIONS: list[str] = [
    "You are worthy of love and belonging.",
    "Your feelings are valid, and it's okay to feel them.",
    "You have the strength to get through this.",
    "You are not alone — support is available.",
    "It's okay to ask for help. That takes courage.",
    "You are doing your best, and that is enough.",
    "Every step forward, no matter how small, is progress.",
    "You are more resilient than you know.",
    "This difficult moment will pass. You will be okay.",
    "You matter — your thoughts, your feelings, your story.",
]


# ---------------------------------------------------------------------------
# Response Engine
# ---------------------------------------------------------------------------

class ResponseEngine:
    """
    Selects an appropriate empathetic response template based on the
    emotion and intensity found in an AnalysisResult.
    """

    def generate(self, result: AnalysisResult) -> str:
        if result.is_crisis:
            return random.choice(RESPONSE_TEMPLATES["crisis"])

        key = self._build_key(result.emotion, result.intensity)
        templates = RESPONSE_TEMPLATES.get(key) or RESPONSE_TEMPLATES.get(result.emotion)

        if not templates:
            templates = RESPONSE_TEMPLATES["neutral"]

        response = random.choice(templates)

        # Append a quote/affirmation for negative high-intensity moods
        if result.sentiment == "negative" and result.intensity > 0.5:
            response += f"\n\n✨ *{random.choice(AFFIRMATIONS)}*"

        return response

    def get_motivational_quote(self) -> str:
        return f"💬 {random.choice(MOTIVATIONAL_QUOTES)}"

    def get_affirmation(self) -> str:
        return f"🌟 {random.choice(AFFIRMATIONS)}"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_key(emotion: Emotion, intensity: float) -> str:
        """
        Maps emotion + intensity to a template pool key.
        Emotions with high/low variants use intensity threshold 0.45.
        """
        INTENSITY_SPLIT = 0.45
        has_variants = {"anxiety", "sadness", "loneliness", "stress", "anger"}

        if emotion in has_variants:
            level = "high" if intensity >= INTENSITY_SPLIT else "low"
            return f"{emotion}_{level}"

        return emotion  # joy, neutral, crisis handled directly
