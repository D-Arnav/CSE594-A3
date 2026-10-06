"""Prompt templates, versioned (SPEC 5.3 / 5.4).

Any change to a prompt = a new version key + an entry in docs/iteration_log.md.
The rank prompt lists the (shuffled) candidates one per line.
"""

PROMPT_RANK = {
    "v1": (
        "You are shown a street-level photo. It was taken in exactly one of these countries:\n"
        "{shuffled_country_list}\n"
        "Which country was it taken in? Answer with the country name only."
    ),
}

# Assistant answer scored for each candidate c (plus the end-of-turn token), keyed by rank version.
# SPEC 5.2 writes "{c}.", but Qwen3-VL-2B answers this prompt with the bare name and gives the
# period log p ~ -20 that varies by 1-2 nats across candidates (pure noise), so v1 scores "{c}".
RANK_ANSWER = {"v1": "{c}"}

PROMPT_EXPLAIN = {
    "v1": (
        "Consider whether this photo could have been taken in {country}.\n"
        "In at most two short sentences, name specific things VISIBLE in the image\n"
        "that support or argue against {country} (e.g., road markings, which side\n"
        "cars drive on, signs and their script, vegetation, architecture, poles).\n"
        "Only mention things you can actually see. If there are no useful clues,\n"
        'say "No distinctive clues."'
    ),
    # v2: v1 gave "No distinctive clues." for 42% / 87% / 99% of top-1/2/3 rationales on dev and
    # opened 56% of top-1 ones with "Yes, this photo could...". Drop the escape hatch and the opener.
    "v2": (
        "Consider whether this street-level photo was taken in {country}.\n"
        "Name one or two specific things you can SEE in the image (road markings, which side cars\n"
        "drive on, signs and their script, vegetation, terrain, architecture, utility poles, vehicles)\n"
        "and say whether each fits {country}. Use at most two short sentences.\n"
        'Start directly with what you see; do not begin with "Yes" or "No".\n'
        "Only mention things that are actually visible."
    ),
    # v3 (rejected, kept for the log): ask for one clue for and one against the country. The model
    # copied the template ("First sentence: ...") in 97.5% of outputs; v2 is used.
    "v3": (
        "Consider whether this street-level photo was taken in {country}.\n"
        "First sentence: one specific thing you can SEE in the image that fits {country}, and why.\n"
        "Second sentence: one specific thing you can SEE that does not fit {country}, or\n"
        '"Nothing visible argues against it."\n'
        'Only mention things that are actually visible. Do not begin with "Yes" or "No".'
    ),
}

# SPEC 5.4 generative baseline (iteration log only). Same candidate list as the rank prompt.
PROMPT_JSON_BASELINE = {
    "v1": (
        "You are shown a street-level photo. It was taken in exactly one of these countries:\n"
        "{shuffled_country_list}\n"
        "Give your top 3 guesses for the country, with a confidence between 0 and 1 for each "
        "and a short reasoning based on what is visible in the photo. "
        "Answer with JSON only, in exactly this format:\n"
        '{{"guesses": [{{"country": "...", "confidence": 0.0, "reasoning": "..."}}, '
        '{{"country": "...", "confidence": 0.0, "reasoning": "..."}}, '
        '{{"country": "...", "confidence": 0.0, "reasoning": "..."}}]}}'
    ),
}


def render_rank(candidates, version="v1"):
    return PROMPT_RANK[version].format(shuffled_country_list="\n".join(candidates))


def render_explain(country, version="v1"):
    return PROMPT_EXPLAIN[version].format(country=country)


def render_json_baseline(candidates, version="v1"):
    return PROMPT_JSON_BASELINE[version].format(shuffled_country_list="\n".join(candidates))
