"""
relaxation_tips.py
------------------
Provides guided relaxation exercises, breathing techniques, and
mindfulness activities tailored to specific emotions.
"""

import random
from .sentiment_analyzer import Emotion


# ---------------------------------------------------------------------------
# Exercise library
# ---------------------------------------------------------------------------

BREATHING_EXERCISES: list[dict] = [
    {
        "name": "4-7-8 Breathing (Calming Breath)",
        "for":  ["anxiety", "stress", "anger"],
        "steps": [
            "Find a comfortable position — sitting upright or lying down.",
            "Close your eyes and relax your jaw.",
            "Inhale quietly through your nose for 4 seconds.",
            "Hold your breath for 7 seconds.",
            "Exhale completely through your mouth for 8 seconds.",
            "This completes one cycle. Repeat 3–4 times.",
        ],
        "benefit": "Activates the parasympathetic nervous system to quickly reduce anxiety and stress.",
    },
    {
        "name": "Box Breathing (Square Breath)",
        "for":  ["stress", "anxiety", "anger"],
        "steps": [
            "Sit with your back straight and feet flat on the floor.",
            "Breathe out slowly, releasing all air from your lungs.",
            "Breathe IN through your nose for 4 counts.",
            "HOLD your breath for 4 counts.",
            "Breathe OUT through your mouth for 4 counts.",
            "HOLD empty for 4 counts.",
            "Repeat this cycle 4 times.",
        ],
        "benefit": "Used by Navy SEALs to stay calm under pressure. Great for acute stress.",
    },
    {
        "name": "Deep Belly Breathing",
        "for":  ["anxiety", "loneliness", "sadness", "neutral"],
        "steps": [
            "Place one hand on your chest and the other on your belly.",
            "Take a slow, deep breath in through your nose (2–3 secs).",
            "Feel your belly — not your chest — rise.",
            "Slowly exhale through pursed lips (2–3 secs).",
            "Feel your belly fall.",
            "Repeat 6–10 times, focusing on each breath.",
        ],
        "benefit": "Reduces cortisol and promotes full-body relaxation.",
    },
    {
        "name": "Alternate Nostril Breathing (Nadi Shodhana)",
        "for":  ["anxiety", "stress"],
        "steps": [
            "Sit comfortably and rest your left hand on your left knee.",
            "Lift your right hand and rest your index and middle fingers between your eyebrows.",
            "Close your RIGHT nostril with your right thumb.",
            "Inhale slowly through your LEFT nostril.",
            "Close your LEFT nostril with your right ring finger.",
            "Open your RIGHT nostril and exhale slowly.",
            "Inhale through your RIGHT nostril.",
            "Close RIGHT and exhale through LEFT.",
            "This is one cycle. Repeat 5–10 times.",
        ],
        "benefit": "Balances the nervous system and calms mental chatter.",
    },
]

GROUNDING_EXERCISES: list[dict] = [
    {
        "name": "5-4-3-2-1 Grounding Technique",
        "for":  ["anxiety", "stress", "anger", "loneliness"],
        "steps": [
            "Pause and look around you. Name 5 things you can SEE.",
            "Name 4 things you can physically TOUCH. Reach out and touch them.",
            "Name 3 things you can HEAR right now.",
            "Name 2 things you can SMELL (or like the smell of).",
            "Name 1 thing you can TASTE.",
            "Take a slow breath. You are present. You are safe.",
        ],
        "benefit": "Interrupts anxiety spirals by anchoring you to the present moment.",
    },
    {
        "name": "Body Scan Relaxation",
        "for":  ["stress", "anxiety", "sadness"],
        "steps": [
            "Lie down or sit comfortably. Close your eyes.",
            "Take 3 slow deep breaths.",
            "Direct your attention to the top of your head. Consciously relax it.",
            "Slowly move attention down: forehead → jaw → neck → shoulders.",
            "Continue: chest → arms → hands → belly → lower back.",
            "Continue: hips → thighs → knees → calves → feet → toes.",
            "At each area, notice any tension and breathe into it.",
            "When you reach your toes, take 3 final deep breaths.",
        ],
        "benefit": "Releases physical tension stored from stress and emotional pain.",
    },
    {
        "name": "Safe Place Visualization",
        "for":  ["loneliness", "sadness", "anxiety"],
        "steps": [
            "Close your eyes and take 3 slow breaths.",
            "Imagine a place where you feel completely safe and calm.",
            "It can be real (a beach, forest, your room) or imaginary.",
            "Visualise the details: colors, sounds, smells, temperature.",
            "Feel yourself fully present in this safe place.",
            "Stay here for 2–5 minutes, breathing slowly.",
            "When ready, open your eyes slowly and bring that calm with you.",
        ],
        "benefit": "Provides emotional refuge during intense sadness or loneliness.",
    },
]

MINDFULNESS_ACTIVITIES: list[dict] = [
    {
        "name": "Mindful Walking (5 minutes)",
        "for":  ["stress", "anger", "neutral"],
        "steps": [
            "Stand up and find a short path (indoors or outdoors).",
            "Walk slowly — about half your normal pace.",
            "Feel each step: heel, arch, toes touching the ground.",
            "Synchronize your breath: inhale for 2 steps, exhale for 2 steps.",
            "When your mind wanders, gently return to the sensation of walking.",
            "Do this for 5 minutes. Notice how you feel afterwards.",
        ],
        "benefit": "Reduces stress hormones and improves mood through mindful movement.",
    },
    {
        "name": "Gratitude Journaling (3-2-1 Method)",
        "for":  ["sadness", "loneliness", "neutral"],
        "steps": [
            "Grab a notebook or open a notes app.",
            "Write down 3 things you are grateful for today (no matter how small).",
            "Write down 2 people who made a positive difference in your life.",
            "Write down 1 thing you are proud of yourself for this week.",
            "Read it back aloud to yourself.",
            "Save it — on hard days, re-read past entries.",
        ],
        "benefit": "Rewires the brain toward positive neural pathways over time.",
    },
    {
        "name": "Progressive Muscle Relaxation",
        "for":  ["stress", "anxiety", "anger"],
        "steps": [
            "Find a quiet place. Sit or lie down comfortably.",
            "Starting with your feet: curl your toes tightly for 5 seconds.",
            "Release and notice the relaxation for 10 seconds.",
            "Move upward: calves → thighs → abs → fists → arms → shoulders → face.",
            "Tense each muscle group for 5 seconds, then release for 10.",
            "Finish with your whole body tensed for 5 seconds, then a full release.",
            "Take 3 deep breaths and enjoy the sensation of relaxation.",
        ],
        "benefit": "Directly dissolves physical tension caused by stress and anxiety.",
    },
    {
        "name": "Digital Detox Mini-Break",
        "for":  ["stress", "anxiety", "neutral"],
        "steps": [
            "Put your phone face-down and close all browser tabs for 15 minutes.",
            "Make yourself a warm drink (tea, water, or coffee).",
            "Sit somewhere comfortable away from your desk.",
            "Do nothing. No phone, no music, no tasks.",
            "If thoughts about work arise, acknowledge them and let them pass.",
            "After 15 minutes, return refreshed with a single priority in mind.",
        ],
        "benefit": "Resets overwhelmed neural pathways and restores focus.",
    },
]


# ---------------------------------------------------------------------------
# Relaxation Tips Engine
# ---------------------------------------------------------------------------

class RelaxationTips:
    """
    Recommends context-appropriate relaxation exercises based on
    the user's detected emotion.
    """

    def get_tip(self, emotion: Emotion) -> str:
        """Return one random exercise most suited to this emotion."""
        all_exercises = BREATHING_EXERCISES + GROUNDING_EXERCISES + MINDFULNESS_ACTIVITIES
        suited = [ex for ex in all_exercises if emotion in ex["for"]]

        if not suited:
            suited = all_exercises  # fallback: any exercise

        exercise = random.choice(suited)
        return self._format_exercise(exercise)

    def get_random_tip(self) -> str:
        """Return a completely random exercise."""
        all_exercises = BREATHING_EXERCISES + GROUNDING_EXERCISES + MINDFULNESS_ACTIVITIES
        return self._format_exercise(random.choice(all_exercises))

    def get_breathing_exercise(self) -> str:
        return self._format_exercise(random.choice(BREATHING_EXERCISES))

    def get_grounding_exercise(self) -> str:
        return self._format_exercise(random.choice(GROUNDING_EXERCISES))

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_exercise(exercise: dict) -> str:
        lines = [
            f"🌿 **{exercise['name']}**",
            "",
        ]
        for i, step in enumerate(exercise["steps"], 1):
            lines.append(f"  {i}. {step}")
        lines += [
            "",
            f"  💡 *Why this helps:* {exercise['benefit']}",
        ]
        return "\n".join(lines)
