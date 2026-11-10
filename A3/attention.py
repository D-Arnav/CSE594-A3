"""Attention checks: two instructed responses, shown on the same screen as a trial."""
from gemini import COUNTRIES

MAX_FAILURES = 2  # tolerant by design: one slip earns a warning, a second ends the session
CHECKS = [
    {"after_trial": 13, "question": "Select South Korea.",
     "options": COUNTRIES, "answer": "South Korea"},
    {"after_trial": 26, "question": "Select Germany.",
     "options": COUNTRIES, "answer": "Germany"},
]


def due_index(state):
    """Index of the check the participant must answer before continuing, or None."""
    index = state["checks_done"]
    return index if index < len(CHECKS) and CHECKS[index]["after_trial"] <= state["done"] else None


def view(index, image):
    """A check as the browser sees it: never includes the answer."""
    return {"index": index, "image": image, "question": CHECKS[index]["question"], "options": CHECKS[index]["options"]}


def is_correct(index, answer):
    return answer == CHECKS[index]["answer"]
