"""Synchronous AI suggestions from the Gemini API."""
import base64
import time
from functools import lru_cache
from typing import Literal, get_args

from google import genai
from pydantic import BaseModel, Field, field_validator

import env  # noqa: F401  (loads .env)

MODEL = "gemini-3.1-flash-lite"  # cheapest free-tier vision model
Country = Literal["United States", "Germany", "Japan", "South Korea", "India"]
COUNTRIES = list(get_args(Country))
ATTEMPTS = 2  # after this the trial falls back to SmolVLM2, so a participant is never stuck
TIMEOUT_MS = 20_000
PROMPT = (f"This street-level photo was taken in exactly one of: {', '.join(COUNTRIES)}. "
          "Return your three most likely countries, most likely first. For each, give the probability it is correct "
          "and one sentence of at most 15 words naming specific things visible in the photo that support it.")


class Suggestion(BaseModel):
    country: Country
    p: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)


class Suggestions(BaseModel):
    suggestions: list[Suggestion]

    @field_validator("suggestions")
    @classmethod
    def three_ranked_countries(cls, value):
        # the API treats list and uniqueness constraints as hints, so enforce them here
        if len(value) != 3 or len({s.country for s in value}) != 3:
            raise ValueError("expected three distinct countries")
        return sorted(value, key=lambda s: -s.p)


class GeminiUnavailable(Exception):
    pass


@lru_cache(maxsize=1)
def client():
    return genai.Client(http_options={"timeout": TIMEOUT_MS})  # reads GEMINI_API_KEY


def suggest(image_path, attempts=ATTEMPTS):
    """-> [{"country", "p", "rationale"}] x 3, ranked. Raises GeminiUnavailable once retries are used up."""
    image = base64.b64encode(image_path.read_bytes()).decode()
    for attempt in range(attempts):
        try:
            interaction = client().interactions.create(
                model=MODEL,
                input=[{"type": "text", "text": PROMPT},
                       {"type": "image", "data": image, "mime_type": "image/jpeg"}],
                response_format={"type": "text", "mime_type": "application/json",
                                 "schema": Suggestions.model_json_schema()})
            parsed = Suggestions.model_validate_json(interaction.output_text)
            return [s.model_dump() for s in parsed.suggestions]
        except Exception as e:  # the SDK raises interactions errors from a private module, and any failure is retryable
            print(f"gemini attempt {attempt + 1}/{attempts} failed for {image_path.name}: {e}", flush=True)
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    raise GeminiUnavailable(image_path.name)
