"""
crisis_handler.py
-----------------
Detects crisis signals and generates appropriate safe-messaging responses
with professional resource referrals.
"""

import random

PROFESSIONAL_RESOURCES: list[str] = [
    "🆘 iCall (India):                 9152987821",
    "🆘 Vandrevala Foundation (India): 1860-2662-345",
    "🆘 NIMHANS (India):               080-46110007",
    "🆘 988 Suicide & Crisis (US):     Call or text 988",
    "🆘 Crisis Text Line (US/UK/CA):   Text HOME to 741741",
    "🆘 Samaritans (UK/Ireland):       116 123",
    "🆘 Lifeline (Australia):          13 11 14",
    "📚 Your university counseling center",
    "👨‍⚕️ A trusted doctor or mental health professional",
]

CRISIS_RESPONSES: list[str] = [
    """\
I'm really glad you reached out right now. What you've shared is important, \
and I want you to know you are not alone.

Please contact one of these resources immediately — real people are ready to help:

{resources}

You matter more than you know. Please make that call or send that text. 💙""",

    """\
Thank you for trusting me with this. I'm taking what you've said very seriously.

You deserve real, professional support right now. Please reach out to one of these helplines:

{resources}

Help is closer than it feels. You are not a burden — you are worthy of care. 🤍""",

    """\
I hear you, and I'm here with you right now. But I also want you to connect with \
someone who can provide real, in-person support.

Please reach out to one of these crisis resources now:

{resources}

One message or one call can change everything. Please try. 💛""",
]


class CrisisHandler:
    """
    Provides crisis-specific responses with professional resource listings.
    """

    def get_response(self) -> str:
        resources = "\n".join(f"  {r}" for r in PROFESSIONAL_RESOURCES)
        template  = random.choice(CRISIS_RESPONSES)
        return template.format(resources=resources)

    @staticmethod
    def get_resources_text() -> str:
        return "\n".join(f"  {r}" for r in PROFESSIONAL_RESOURCES)
