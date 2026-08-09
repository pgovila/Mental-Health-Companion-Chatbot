"""
main.py
-------
Entry point for the Mental Health Companion Chatbot.
Provides a rich, colour-coded CLI experience using the `rich` library.
"""

import sys
import os

# Force UTF-8 on Windows so Rich can render box-drawing chars safely.
# Must happen before any other import that touches stdout.
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Ensure the project root is on the path when run directly
sys.path.insert(0, os.path.dirname(__file__))

try:
    from rich.console import Console
    from rich.panel   import Panel
    from rich.prompt  import Prompt
except ImportError:
    print("Missing dependencies. Please run:  pip install -r requirements.txt")
    sys.exit(1)

from modules.chatbot import MentalHealthChatbot
from modules.sentiment_analyzer import AnalysisResult

# ---------------------------------------------------------------------------
# Console  (force_terminal=True so Rich uses its full renderer on Windows)
# ---------------------------------------------------------------------------

console = Console(force_terminal=True, highlight=False)

# ---------------------------------------------------------------------------
# Mood display helpers  (ASCII-safe — no emoji in labels)
# ---------------------------------------------------------------------------

EMOTION_LABEL: dict[str, str] = {
    "anxiety":    "[ANX]",
    "sadness":    "[SAD]",
    "loneliness": "[LON]",
    "stress":     "[STR]",
    "anger":      "[ANG]",
    "joy":        "[JOY]",
    "neutral":    "[NEU]",
}

SENTIMENT_COLOR: dict[str, str] = {
    "positive": "green",
    "negative": "red",
    "neutral":  "yellow",
}

EMOTION_COLOR: dict[str, str] = {
    "anxiety":    "yellow",
    "sadness":    "blue",
    "loneliness": "cyan",
    "stress":     "orange3",
    "anger":      "red",
    "joy":        "green",
    "neutral":    "white",
}


# ---------------------------------------------------------------------------
# Emoji stripping helper
# ---------------------------------------------------------------------------

import re

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # supplemental symbols (brain, etc.)
    "\U00002600-\U000026FF"  # misc symbols
    "\u2640-\u2642"
    "\u2194-\u2199"
    "\u23cf\u23e9\u231a"
    "\ufe0f"                 # variation selector
    "\u20d0-\u20ff"          # combining marks
    "]+",
    flags=re.UNICODE,
)

def _safe(text: str) -> str:
    """Strip emoji so Windows legacy console never chokes."""
    return _EMOJI_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def print_welcome() -> None:
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Mental Health Companion[/bold cyan]\n"
        "[dim]A safe space for students -- powered by AI & NLP[/dim]\n\n"
        "[white]I'm here to listen, support, and guide you.[/white]\n"
        "[dim]Share how you're feeling, or type [bold]help[/bold] to see what I can do.[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))
    console.print()


def print_disclaimer() -> None:
    console.print(Panel(
        "[bold yellow]Important Disclaimer[/bold yellow]\n\n"
        "This chatbot is a [bold]supportive tool[/bold], not a replacement for professional\n"
        "mental health care. If you are in crisis or feel unsafe, please\n"
        "contact a counselor, helpline, or emergency services immediately.",
        border_style="yellow",
        padding=(0, 2),
    ))
    console.print()


def print_mood_badge(analysis: AnalysisResult) -> None:
    """Print a compact mood indicator line after each user message."""
    emotion = analysis.emotion
    label   = EMOTION_LABEL.get(emotion, "[?]")
    e_color = EMOTION_COLOR.get(emotion, "white")
    s_color = SENTIMENT_COLOR.get(analysis.sentiment, "white")
    bar     = _polarity_bar(analysis.polarity)

    console.print(
        f"  [dim]Detected:[/dim] "
        f"[{e_color}]{label} {emotion.upper()}[/{e_color}]  "
        f"[{s_color}]({analysis.sentiment})[/{s_color}]  "
        f"[dim]{bar}  intensity {analysis.intensity:.0%}[/dim]"
    )


def _polarity_bar(polarity: float) -> str:
    """Visual polarity bar using standard ASCII chars."""
    filled = int((polarity + 1.0) / 2.0 * 10)
    filled = max(0, min(10, filled))
    return f"[{'#' * filled}{'-' * (10 - filled)}]"


def print_bot_response(text: str, is_crisis: bool = False) -> None:
    border = "red" if is_crisis else "cyan"
    prefix = "[bold red]!! CRISIS SUPPORT[/bold red]" if is_crisis else "[bold cyan]>> Companion[/bold cyan]"
    clean  = _safe(text)

    console.print()
    console.print(Panel(
        f"{prefix}\n\n{clean}",
        border_style=border,
        padding=(0, 2),
    ))


def print_tip(tip_text: str) -> None:
    console.print()
    console.print(Panel(
        f"[bold green]>> Relaxation Exercise[/bold green]\n\n{_safe(tip_text)}",
        border_style="green",
        padding=(0, 2),
    ))


def print_farewell(farewell_text: str, mood_summary: str, ascii_chart: str) -> None:
    console.print()
    console.print(Panel(
        f"[bold cyan]Goodbye![/bold cyan]\n\n{_safe(farewell_text)}",
        border_style="cyan",
        padding=(0, 2),
    ))
    if mood_summary:
        console.print()
        console.print(Panel(
            _safe(mood_summary),
            title="[bold]Session Report[/bold]",
            border_style="blue",
            padding=(0, 2),
        ))
    if ascii_chart:
        console.print()
        console.print(Panel(
            ascii_chart,
            title="[bold]Polarity Chart[/bold]",
            border_style="dim",
            padding=(0, 2),
        ))


def print_generic_response(text: str) -> None:
    """For command responses (mood, relax, quote, help)."""
    console.print()
    console.print(Panel(
        _safe(text),
        border_style="cyan",
        padding=(0, 2),
    ))


def offer_tip_prompt(emotion: str) -> bool:
    """Ask the user if they want a relaxation exercise."""
    try:
        answer = Prompt.ask(
            f"\n  [dim]Would you like a relaxation exercise for {emotion}? [y/n][/dim]",
            choices=["y", "n", "yes", "no"],
            default="y",
            show_choices=False,
        ).lower().strip()
        return answer in ("y", "yes")
    except (KeyboardInterrupt, EOFError):
        return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run() -> None:
    print_welcome()
    print_disclaimer()

    bot = MentalHealthChatbot()

    console.print("[dim]  Type your message below and press Enter. Type [bold]quit[/bold] to exit.[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold white]  You[/bold white]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n")
            response = bot.respond("quit")
            print_farewell(response.text, response.mood_summary or "", response.ascii_chart or "")
            break

        if not user_input:
            console.print("  [dim]Please type something -- I'm listening.[/dim]")
            continue

        lower = user_input.lower()

        # ── Process ──────────────────────────────────────────────────────
        response = bot.respond(user_input)

        # ── Farewell ─────────────────────────────────────────────────────
        if response.is_farewell:
            print_farewell(response.text, response.mood_summary or "", response.ascii_chart or "")
            break

        # ── Mood badge (only for real conversational turns) ───────────────
        if response.analysis:
            print_mood_badge(response.analysis)

        # ── Main response ─────────────────────────────────────────────────
        if lower in ("mood", "relax", "quote", "help"):
            print_generic_response(response.text)
        else:
            print_bot_response(response.text, is_crisis=response.is_crisis)

        # ── Relaxation tip ────────────────────────────────────────────────
        if response.show_tip and response.tip_text:
            if offer_tip_prompt(response.analysis.emotion if response.analysis else "stress"):
                print_tip(response.tip_text)

        console.print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.getLogger("nltk").setLevel(logging.ERROR)

    # Ensure TextBlob corpora are available
    try:
        import nltk
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        console.print("[dim]  Downloading required language data (first run only)...[/dim]")
        import nltk as _nltk
        _nltk.download("punkt", quiet=True)
        _nltk.download("averaged_perceptron_tagger", quiet=True)

    run()
