"""Geolocation study: participants label random photos, with or without AI suggestions.

In the AI condition, a fixed share of each session's trials calls Gemini live; the rest show SmolVLM2's suggestions
from Assignment 2. The source of every trial is logged."""
import csv
import random
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import attention
import db
import gemini
import smolvlm

ROOT = Path(__file__).parent
IMAGE_DIR = ROOT / "images"
TRIALS_PER_COUNTRY = 8
GEMINI_SHARE = 0.2  # of each AI session's trials
UNIQNAME = re.compile(r"[a-z]{3,8}")  # U-M uniqnames are 3 to 8 letters
N_TRIALS = TRIALS_PER_COUNTRY * len(gemini.COUNTRIES)
N_GEMINI = round(N_TRIALS * GEMINI_SHARE)
with open(ROOT / "images.csv", newline="") as f:
    TRUTH = {row["image_id"]: row["country"] for row in csv.DictReader(f)}
BY_COUNTRY = {c: sorted(i for i, country in TRUTH.items() if country == c) for c in gemini.COUNTRIES}


@asynccontextmanager
async def lifespan(_):
    db.init()
    yield


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")


class NewSession(BaseModel):
    ai_assist: bool
    uniqname: str = Field(max_length=100)

    @field_validator("uniqname")
    @classmethod
    def normalize_uniqname(cls, value):
        value = value.strip().lower().removesuffix("@umich.edu")
        if not UNIQNAME.fullmatch(value):
            raise ValueError("a uniqname is 3 to 8 letters")
        return value


class CheckAnswer(BaseModel):
    check_index: int = Field(ge=0)
    answer: str = Field(max_length=200)


class Answer(BaseModel):
    trial_index: int = Field(ge=0)
    response: str
    expanded_other: bool = False
    rt_ms: int = Field(ge=0, le=3_600_000)


class Survey(BaseModel):
    confidence: int = Field(ge=1, le=7)
    mental_demand: int = Field(ge=1, le=7)
    geo_knowledge: int = Field(ge=1, le=7)
    seen_other_condition: bool
    trust: int | None = Field(None, ge=1, le=7)
    usefulness: int | None = Field(None, ge=1, le=7)
    free_text: str = Field("", max_length=5000)


ASSIGNMENTS = {}  # participant_id -> photos; fixed once stored, so caching them in memory is safe


def assignment(participant_id):
    """The session's photos, drawn once when it starts and stored, so resuming continues the same list."""
    if participant_id not in ASSIGNMENTS:
        ASSIGNMENTS[participant_id] = db.get_assignment(participant_id) or db.save_assignment(
            participant_id, *draw_photos(participant_id))
    return ASSIGNMENTS[participant_id]


def draw_photos(participant_id):
    """-> (trial photos, attention-check photos): a random 8 per country, shuffled, plus one photo per check.
    Photos from the same person's other sessions are excluded, so no photo appears in both conditions."""
    uniqname, condition = participant_id.split("-")[:2]
    others = db.other_assignments(uniqname, participant_id)
    seen = {i for _, a in others for i in a["images"] + a["check_images"]}
    other_condition = {i for pid, a in others if pid.split("-")[1] != condition for i in a["images"] + a["check_images"]}
    rng = random.Random(participant_id)
    images = []
    for country in gemini.COUNTRIES:
        pool = [i for i in BY_COUNTRY[country] if i not in seen]
        if len(pool) < TRIALS_PER_COUNTRY:
            # ponytail: after several restarts a photo may repeat from an earlier attempt of this condition, never
            # from the other condition; past ~4 restarts in both conditions a country runs out and this raises
            pool = [i for i in BY_COUNTRY[country] if i not in other_condition]
        images += rng.sample(pool, TRIALS_PER_COUNTRY)
    rng.shuffle(images)
    rest = sorted(set(TRUTH) - set(images) - seen) or sorted(set(TRUTH) - set(images) - other_condition)
    return images, rng.sample(rest, len(attention.CHECKS))


def image_order(participant_id):
    return assignment(participant_id)["images"]


def gemini_trials(participant_id):
    """Which trial indices call Gemini live: a random set of N_GEMINI, seeded by id so a refresh keeps it."""
    return set(random.Random(f"{participant_id}-gemini").sample(range(N_TRIALS), N_GEMINI))


def image_url(image_id):
    return f"/images/{image_id}.jpg"


def attention_image(participant_id, index):
    """A photo outside all of this person's trials, so a check adds no exposure to a scored photo."""
    return assignment(participant_id)["check_images"][index]


def participant(participant_id):
    found = db.get_participant(participant_id)
    if not found:
        raise HTTPException(404, "unknown participant")
    return {**found, "terminated": found["check_failures"] >= attention.MAX_FAILURES}


def active_participant(participant_id):
    state = participant(participant_id)
    if state["terminated"]:
        raise HTTPException(409, "session ended by attention checks")
    return state


def session_view(participant_id):
    """Everything the browser needs to decide its next screen. The server owns the flow."""
    state = participant(participant_id)
    due = attention.due_index(state)
    return {"participant_id": participant_id, **state, "countries": gemini.COUNTRIES,
            "attention_check": None if due is None else attention.view(due, image_url(attention_image(participant_id, due))),
            "images": [image_url(image_id) for image_id in image_order(participant_id)]}


@app.get("/")
def index():
    return FileResponse(ROOT / "static/index.html")


@app.post("/api/sessions")
def create_session(req: NewSession):
    """Resume this uniqname's session in this condition, or start one. The id is uniqname plus condition, e.g.
    jsmith-ai; a session ended by attention checks makes the next attempt jsmith-ai-2, and so on."""
    base = f"{req.uniqname}-{'ai' if req.ai_assist else 'noai'}"
    attempt = 1
    while True:
        participant_id = base if attempt == 1 else f"{base}-{attempt}"
        found = db.get_participant(participant_id)
        if found is None:
            db.add_participant(participant_id, req.ai_assist)
            return session_view(participant_id)
        if found["check_failures"] < attention.MAX_FAILURES:
            return session_view(participant_id)
        attempt += 1


@app.get("/api/sessions/{participant_id}")
def get_session(participant_id: str):
    return session_view(participant_id)


@app.post("/api/sessions/{participant_id}/attention")
def submit_attention(participant_id: str, req: CheckAnswer):
    state = participant(participant_id)
    if req.check_index < state["checks_done"]:  # a retry of a saved answer
        return {**session_view(participant_id), "passed": None}
    if state["terminated"] or req.check_index != attention.due_index(state):
        raise HTTPException(409, "this check is not due")
    if req.answer not in attention.CHECKS[req.check_index]["options"]:
        raise HTTPException(422, "answer is not one of the options")
    passed = attention.is_correct(req.check_index, req.answer)
    db.add_attention_check(participant_id, req.check_index, req.answer, passed)
    return {**session_view(participant_id), "passed": passed}


@app.get("/api/sessions/{participant_id}/ai/{trial_index}")
def get_ai(participant_id: str, trial_index: int):
    if not active_participant(participant_id)["ai_assist"]:
        raise HTTPException(403, "this participant is in the no-AI condition")
    order = image_order(participant_id)
    if not 0 <= trial_index < len(order):
        raise HTTPException(404, "no such trial")
    shown = db.get_ai_shown(participant_id, trial_index)
    if shown:
        return shown["suggestions"]
    image_id = order[trial_index]
    source, suggestions = "smolvlm", smolvlm.suggest(image_id)
    if trial_index in gemini_trials(participant_id):
        try:
            source, suggestions = "gemini", gemini.suggest(IMAGE_DIR / f"{image_id}.jpg")
        except gemini.GeminiUnavailable:
            source = "smolvlm_fallback"  # Gemini failed after retries; keep the participant moving and log it
    return db.save_ai_shown(participant_id, trial_index, image_id, source, suggestions)["suggestions"]


@app.post("/api/sessions/{participant_id}/responses")
def submit_response(participant_id: str, req: Answer):
    state = participant(participant_id)
    if req.trial_index < state["done"]:  # a retry of a saved answer
        return session_view(participant_id)
    if state["terminated"] or attention.due_index(state) is not None:
        raise HTTPException(409, "an attention check comes first")
    if req.trial_index > state["done"] or req.trial_index >= N_TRIALS:
        raise HTTPException(409, f"expected trial {state['done']}")
    if req.response not in gemini.COUNTRIES:
        raise HTTPException(422, "response is not one of the countries")
    image_id = image_order(participant_id)[req.trial_index]
    truth = TRUTH[image_id]
    chose_rank = ai_source = ai_top1 = ai_top1_correct = None
    if state["ai_assist"]:
        shown = db.get_ai_shown(participant_id, req.trial_index)
        if not shown:
            raise HTTPException(409, "AI suggestions were not shown for this trial")
        ai_source = shown["source"]
        ranked = [s["country"] for s in shown["suggestions"]]
        chose_rank = ranked.index(req.response) + 1 if req.response in ranked else 0
        if chose_rank == 0 and not req.expanded_other:
            raise HTTPException(422, "choosing outside the AI's suggestions requires 'None of these'")
        ai_top1, ai_top1_correct = ranked[0], int(ranked[0] == truth)
    elif req.expanded_other:
        raise HTTPException(422, "the no-AI condition has no 'None of these'")
    db.add_trial(participant_id, req.trial_index, image_id, state["ai_assist"], req.response, req.response == truth,
                 chose_rank, req.expanded_other, ai_source, ai_top1, ai_top1_correct, req.rt_ms)
    return session_view(participant_id)


@app.post("/api/sessions/{participant_id}/survey")
def submit_survey(participant_id: str, req: Survey):
    state = active_participant(participant_id)
    if state["done"] < N_TRIALS:
        raise HTTPException(409, "trials not finished")
    if state["ai_assist"] != (req.trust is not None) or state["ai_assist"] != (req.usefulness is not None):
        raise HTTPException(422, "trust and usefulness are asked only in the AI condition")
    db.add_survey(participant_id, req.confidence, req.mental_demand, req.geo_knowledge, req.seen_other_condition,
                  req.trust, req.usefulness, req.free_text)
    return session_view(participant_id)
