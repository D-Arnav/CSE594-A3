"""Synchronous AI suggestions from the Gemini API."""
import base64
import io
import time
from functools import lru_cache
from typing import Literal, get_args

from google import genai
from PIL import Image
from pydantic import BaseModel, Field, field_validator

import env  # noqa: F401  (loads .env)

MODEL = "gemini-3.1-flash-lite"  # cheapest free-tier vision model; the study does not need a strong one
DOWNSCALE = 8  # the AI sees photos at 1/8 width and height, which handicaps it; participants see the originals
MODEL_LABEL = f"{MODEL} at 1/{DOWNSCALE} scale"
Country = Literal["United States", "Germany", "Japan", "South Korea", "India"]
COUNTRIES = list(get_args(Country))
ATTEMPTS = 3
TIMEOUT_MS = 30_000
PROMPT = (f"This street-level photo was taken in exactly one of: {', '.join(COUNTRIES)}. "
          "Return your three most likely countries, most likely first. For each, give the probability it is correct "
          "and one short sentence naming specific things visible in the photo that support it.")


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


def downscaled_jpeg(image_path):
    with Image.open(image_path) as photo:
        size = (max(1, photo.width // DOWNSCALE), max(1, photo.height // DOWNSCALE))
        small = photo.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    small.save(buffer, "JPEG", quality=90)
    return buffer.getvalue()


def suggest(image_path, attempts=ATTEMPTS):
    """-> [{"country", "p", "rationale"}] x 3, ranked. Raises GeminiUnavailable once retries are used up."""
    image = base64.b64encode(downscaled_jpeg(image_path)).decode()
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
