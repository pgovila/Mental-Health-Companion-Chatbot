"""
app.py
------
Flask web server for the Mental Health Companion Chatbot.
Exposes a single-page chat UI and a JSON API endpoint.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify, session
from modules.chatbot import MentalHealthChatbot
from modules.relaxation_tips import RelaxationTips
from modules.response_engine import ResponseEngine
import uuid

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "mhc-dev-secret-2024")

# ---------------------------------------------------------------------------
# Per-session chatbot instances (keyed by session ID)
# ---------------------------------------------------------------------------

_sessions: dict[str, MentalHealthChatbot] = {}

def _get_bot() -> MentalHealthChatbot:
    """Return (or create) the chatbot instance for this browser session."""
    sid = session.get("sid")
    if not sid or sid not in _sessions:
        sid = str(uuid.uuid4())
        session["sid"] = sid
        _sessions[sid] = MentalHealthChatbot()
    return _sessions[sid]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "empty message"}), 400

    bot = _get_bot()
    resp = bot.respond(user_message)

    # Build structured payload for the frontend
    payload = {
        "text":       _strip_markdown_bold(resp.text),
        "is_crisis":  resp.is_crisis,
        "is_farewell": resp.is_farewell,
        "tip":        _strip_markdown_bold(resp.tip_text) if resp.show_tip and resp.tip_text else None,
        "analysis":   None,
        "summary":    None,
    }

    if resp.analysis:
        payload["analysis"] = {
            "emotion":    resp.analysis.emotion,
            "sentiment":  resp.analysis.sentiment,
            "polarity":   resp.analysis.polarity,
            "intensity":  round(resp.analysis.intensity * 100),
        }

    if resp.is_farewell and resp.mood_summary:
        payload["summary"] = resp.mood_summary

    return jsonify(payload)


@app.route("/api/reset", methods=["POST"])
def reset():
    """Clear the current session and start a fresh chatbot."""
    sid = session.get("sid")
    if sid and sid in _sessions:
        del _sessions[sid]
    session.pop("sid", None)
    return jsonify({"ok": True})


@app.route("/api/relax", methods=["GET"])
def relax():
    tips = RelaxationTips()
    return jsonify({"tip": _strip_markdown_bold(tips.get_random_tip())})


@app.route("/api/quote", methods=["GET"])
def quote():
    engine = ResponseEngine()
    return jsonify({"quote": engine.get_motivational_quote()})


@app.route("/debug-hf", methods=["GET"])
def debug_hf():
    """
    TEMPORARY diagnostic route — visit this URL in your browser to see
    exactly what happens when the app tries to reach the HuggingFace
    Inference API, without our normal error-swallowing fallback hiding it.
    Reveals only whether the token is present and its length (never the
    token itself), plus the real success/error result of a test API call.
    Remove this route once the issue is diagnosed and fixed — it's not
    meant to stay in a production deployment long-term.
    """
    token = os.environ.get("HF_API_TOKEN")
    result = {
        "token_present": bool(token),
        "token_length":  len(token) if token else 0,
    }
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=token)
        raw = client.text_classification(
            "I am so happy today",
            model="j-hartmann/emotion-english-distilroberta-base",
        )
        result["success"] = True
        result["classification_result"] = [
            {"label": r.label, "score": r.score} for r in raw
        ]
    except Exception as e:
        result["success"] = False
        result["error_type"] = type(e).__name__
        result["error_message"] = str(e)

    return jsonify(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import re

def _strip_markdown_bold(text: str) -> str:
    """Convert **bold** markdown to plain text for the web renderer."""
    if not text:
        return text
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    return text


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n  Mental Health Companion — Web Interface")
    print("  ----------------------------------------")
    print("  Open your browser at:  http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000)