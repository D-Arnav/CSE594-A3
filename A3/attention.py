"""Attention checks: two instructed responses during the task."""
from gemini import COUNTRIES

MAX_FAILURES = 2  # tolerant by design: one slip earns a warning, a second ends the session
CHECKS = [
    {"after_trial": 7, "question": "To show you are paying attention, please select South Korea.",
     "options": COUNTRIES, "answer": "South Korea"},
    {"after_trial": 14, "question": "To show you are paying attention, please select Germany.",
     "options": COUNTRIES, "answer": "Germany"},
]


def due_index(state):
    """Index of the check the participant must answer before continuing, or None."""
    index = state["checks_done"]
    return index if index < len(CHECKS) and CHECKS[index]["after_trial"] <= state["done"] else None


def view(index):
    """A check as the browser sees it: never includes the answer."""
    return {"index": index, "question": CHECKS[index]["question"], "options": CHECKS[index]["options"]}


def is_correct(index, answer):
    return answer == CHECKS[index]["answer"]
