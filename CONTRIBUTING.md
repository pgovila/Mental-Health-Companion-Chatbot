# Contributing

Thanks for your interest in this project. It started as a portfolio/learning
build, so contributions are welcome but please keep a few things in mind
given the subject matter.

## Ground rules

- **Safety first.** Any change touching `modules/crisis_handler.py` or the
  crisis-phrase detection in `modules/sentiment_analyzer.py` needs extra care.
  If you're adding or removing crisis phrases, explain your reasoning in the
  PR description — false negatives here are the one failure mode this project
  can't tolerate.
- **No unverified clinical claims.** Please don't add language implying the
  bot diagnoses conditions, replaces therapy, or has been clinically validated
  — see the Limitations section in the README for why.
- **Keep dependencies lean where possible.** This project targets free-tier
  hosting (Render's 512MB RAM free instances). Before adding a new heavy
  dependency, consider whether a lighter alternative exists.

## Getting started

1. Fork the repo and clone your fork.
2. `pip install -r requirements.txt` (see the README for the CPU-only torch
   install note if you're on a constrained machine).
3. Run `python main.py` (CLI) or `python app.py` (web) to test your changes
   locally before opening a PR.

## Pull requests

- Keep PRs focused — one feature or fix per PR is easier to review.
- Describe what changed and why, especially for anything touching emotion
  classification or response generation, since those are the parts most
  likely to affect how the bot "feels" to someone using it while distressed.

## Reporting issues

If you find a case where the bot responds inappropriately to a message
indicating real distress, please open an issue (or contact the maintainer
directly if you'd rather not post the message content publicly).
