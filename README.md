# 🧠 Mental Health Companion Chatbot

A safe, AI-driven chatbot for students that detects user mood through sentiment analysis and generates empathetic, motivational responses along with relaxation tips to support student mental well-being.

Available in two modes: a **rich terminal CLI** and a **full web interface**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Features

- **Sentiment Analysis** — Detects mood (positive, negative, neutral) from your text using TextBlob NLP
- **Emotion Classification** — Identifies 7 specific emotions: anxiety, sadness, loneliness, stress, anger, joy, neutral
- **Empathetic Responses** — 14 template pools (per emotion × intensity level) for proportionate, warm replies
- **Motivational Quotes** — 15 curated uplifting quotes delivered on demand or during high-distress turns
- **Relaxation Exercises** — 11 practical exercises: breathing techniques, grounding, and mindfulness activities
- **Session Mood Tracker** — Tracks mood history across the session with trend detection (improving / declining / stable)
- **ASCII Polarity Chart** — Visual mood-over-time chart shown at session end (CLI) or in sidebar (Web)
- **Crisis Detection** — Gently redirects to professional resources when serious distress phrases are detected
- **Web Interface** — Full single-page chat UI with live mood sidebar, polarity bar, and emotion badge
- **REST API** — JSON endpoints usable from any frontend or tool

---

## Project Structure

```
mental_health_chatbot/
├── main.py                     # CLI entry point (Rich terminal interface)
├── app.py                      # Web entry point (Flask server)
├── requirements.txt            # Python dependencies
├── templates/
│   └── index.html              # Single-page chat UI
├── static/
│   ├── style.css               # Dark-theme responsive stylesheet
│   └── chat.js                 # Frontend logic (fetch API, DOM, sidebar)
├── modules/
│   ├── __init__.py
│   ├── sentiment_analyzer.py   # TextBlob polarity + keyword emotion engine
│   ├── response_engine.py      # Response templates, quotes, affirmations
│   ├── relaxation_tips.py      # Breathing, grounding, mindfulness exercises
│   ├── mood_tracker.py         # Session history, trend, ASCII chart
│   ├── crisis_handler.py       # Crisis detection & helpline referrals
│   └── chatbot.py              # Core orchestration (used by both CLI & Web)
└── README.md
```

---

## Installation

```bash
cd mental_health_chatbot
pip install -r requirements.txt
```

> NLTK language data is downloaded automatically on first run.

---

## Running — CLI Mode

```bash
python main.py
```

A Rich terminal UI launches with colour-coded mood badges, polarity bars, and styled panels.

### CLI Commands

| Command       | Description                                    |
|---------------|------------------------------------------------|
| `mood`        | Show your current session mood chart           |
| `relax`       | Get a relaxation exercise                      |
| `quote`       | Get a motivational quote                       |
| `help`        | Show available commands                        |
| `quit` / `bye`| End session and display mood summary report    |

---

## Running — Web Mode

```bash
python app.py
```

Then open your browser at **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## Deploying to Render (free tier)

1. Push this repo to GitHub (including `Procfile`, `render.yaml`, and the updated `requirements.txt` — but **not** `venv/`, which `.gitignore` now excludes).
2. On [Render](https://render.com), click **New → Blueprint**, connect this repo, and Render will read `render.yaml` automatically.
3. Alternatively, create the service manually: **New → Web Service**, build command `pip install -r requirements.txt`, start command `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`.
4. First deploy will take a few minutes longer than usual — it's downloading the ~330MB emotion-classification model and CPU-only PyTorch build.

> **A note on the free tier:** Render's free instances have 512MB RAM. The transformer model and PyTorch runtime together typically use 300–450MB once loaded, which is close to that ceiling. If the service crashes or restarts under load, either upgrade to a paid instance, or fall back to keyword-only mode by removing `transformers`/`torch` from `requirements.txt` — the analyzer is written to degrade gracefully to keyword-based classification if the transformer model can't load, so the app keeps working either way.

### Web Interface Features

- **Chat panel** — Full-width conversation view with user and bot bubbles
- **Mood sidebar** — Live positive / negative / neutral message counts
- **Polarity bar** — Smoothly animated session-average mood meter (😞 → 😊)
- **Emotion badge** — Shows the last detected emotion, colour-coded per type
- **Quick-action buttons** — One-click access to Quote, Relax, and Mood Summary
- **Crisis styling** — Red-bordered alert bubble with helpline numbers
- **New Session button** — Resets the conversation and all sidebar counters
- **Responsive design** — Sidebar collapses on mobile screens

### REST API Endpoints

| Method | Endpoint      | Description                                     |
|--------|---------------|-------------------------------------------------|
| `GET`  | `/`           | Serves the single-page chat interface           |
| `POST` | `/api/chat`   | Send a message, receive analysis + response     |
| `GET`  | `/api/quote`  | Returns a random motivational quote             |
| `GET`  | `/api/relax`  | Returns a random relaxation exercise            |
| `POST` | `/api/reset`  | Clears the current session and starts fresh     |

#### `/api/chat` — Request

```json
{ "message": "I feel really anxious about my exams" }
```

#### `/api/chat` — Response

```json
{
  "text": "I can sense the anxiety in your words...",
  "is_crisis": false,
  "is_farewell": false,
  "tip": "4-7-8 Breathing\n  1. Inhale for 4 seconds...",
  "analysis": {
    "emotion": "anxiety",
    "sentiment": "negative",
    "polarity": -0.25,
    "intensity": 46
  },
  "summary": null
}
```

---

## AI / NLP Pipeline

```
User Input
  → Text Cleaning (lowercase, strip punctuation)
  → TextBlob Polarity Score       (-1.0 → +1.0)
  → Emotion Classification:
      • HuggingFace transformer (j-hartmann/emotion-english-distilroberta-base)
        classifies into 7 general emotions (anger, disgust, fear, joy,
        neutral, sadness, surprise), mapped onto this project's taxonomy
      • Keyword lexicon (7 domain emotions × multi-word phrases) fills in
        "stress" and "loneliness", which the transformer's label set doesn't
        cover, and acts as an offline fallback if the model can't load
  → Intensity Scoring             (polarity + subjectivity + keyword density + model confidence)
  → Crisis Phrase Matching        (independent of the above — always runs)
  → Response Template Selection   (14 pools by emotion × intensity)
  → Relaxation Tip Matching       (emotion-tagged exercise library)
  → Mood Tracker Recording
```

**Why hybrid instead of pure transformer?** The transformer model is a general-purpose emotion classifier — it wasn't fine-tuned on student mental-health language, and it has no label for "stress" or "loneliness" specifically, both of which matter a lot in this context (exam pressure, isolation). Combining it with a targeted keyword lexicon gets the best of both: broader, more accurate language understanding from the model, plus precision on the two domain-specific categories the model can't express. Crisis-phrase detection deliberately stays a separate, always-on keyword check rather than depending on either classifier — a false negative there is the one failure mode this project can't tolerate.

---

## Relaxation Exercise Library

| Category | Exercises |
|----------|-----------|
| Breathing | 4-7-8 Breathing, Box Breathing, Deep Belly Breathing, Alternate Nostril |
| Grounding | 5-4-3-2-1 Sensory, Body Scan Relaxation, Safe Place Visualization |
| Mindfulness | Mindful Walking, Gratitude Journaling (3-2-1), Progressive Muscle Relaxation, Digital Detox Break |

---

## Crisis Resources

When crisis-level distress is detected, the bot immediately surfaces:

| Region | Resource |
|--------|----------|
| India | iCall: 9152987821 · Vandrevala: 1860-2662-345 · NIMHANS: 080-46110007 |
| US | 988 Suicide & Crisis Lifeline · Crisis Text Line: Text HOME to 741741 |
| UK / Ireland | Samaritans: 116 123 |
| Australia | Lifeline: 13 11 14 |

---

## Limitations

This project is a portfolio/learning build, not a clinically validated tool. Being transparent about its limits:

- **No clinical validation.** The emotion classifier and response templates have not been evaluated against any clinical benchmark or reviewed by a mental health professional. Accuracy on real distressed speech (as opposed to clearly-worded test sentences) is unverified.
- **Crisis detection is keyword-based.** It catches explicit phrases (e.g. "kill myself," "end it all") but will miss indirect, coded, or non-English expressions of suicidal ideation. It is a safety net, not a substitute for trained crisis intervention.
- **No memory across sessions.** Each session's mood history lives only in server memory (`_sessions` dict in `app.py`) and is lost on restart or after a period of inactivity — there's no persistent user history, journaling, or long-term trend tracking yet.
- **English only.** Both TextBlob and the transformer model are English-language tools; input in other languages will produce unreliable sentiment/emotion readings.
- **General-purpose emotion model.** The HuggingFace model wasn't fine-tuned on student mental-health or counseling-specific text, so its confidence scores may not perfectly reflect real-world severity.
- **Response generation is template-based, not generative.** Replies are selected from a fixed pool per emotion/intensity combination, not written fresh per conversation — so multi-turn context and nuance are limited.

## Data & Privacy

- Chat messages are processed in-memory to generate a response and are **not written to a database or log file** by default.
- Each browser gets a random session ID (via Flask's signed session cookie) used only to keep that browser's `MentalHealthChatbot` instance separate from others on the same server — it is not tied to any account or identity.
- Session data (mood history, message snippets) is held in server RAM only, for the lifetime of that process, and is cleared on server restart or when `/api/reset` is called.
- No message content is sent to any third party. The transformer model runs locally within the app's own process, so message text never leaves the server — the only outbound calls this project makes are the one-time model download from HuggingFace's model hub during install.
- If you deploy this publicly and plan to add persistent history (e.g. a database), disclose that clearly to users before storing anything.

## Disclaimer

This chatbot is a **supportive tool only**, not a replacement for professional mental health care.  
If you or someone you know is in crisis, please contact a qualified counselor, helpline, or emergency services immediately.
