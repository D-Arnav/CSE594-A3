"""Trial allocation for the Assignment 3 study (SPEC 11.2).

Sets: the 200 study trials are sorted by (ai_correct_top1, cluster, country,
seeded tiebreak) and dealt round-robin into n_sets = 200 // block_size sets,
so every set gets an even share of AI-correct/incorrect trials and a spread
of clusters/countries (each country spread across sets). With
block_size 6 that is 33 sets and the 2 left-over trials are never shown.

Rotation (cycle = 2 * n_sets participants, i.e. 50 for blocks of 8):
participant p has pair k = p // 2 within the cycle and

    first block  = set k               second block = set (k + d) % n_sets

with d = 1 + (cycle number) % (n_sets - 1), so the pairing of sets changes
from one cycle to the next. Even p does human-only first, odd p human-AI
first. Over each complete cycle every set fills each of the four slots
(even/first = human-only, even/second = human-AI, odd/first = human-AI,
odd/second = human-only) exactly once, so it appears twice per condition and
twice per block position. Because participants 2k and 2k+1 get the same two
sets with conditions swapped, conditions are also balanced per image after
every even number of participants, not only at full cycles.

Practice: one dev-split trial per condition (never in the 200 study trials),
chosen with the seed and shown before Block A in the participant's block order.
"""
import random

HUMAN, HUMAN_AI = "human_only", "human_ai"
BLOCK_SIZE = 8  # SPEC 11.2; pilot fallback is 6 (config study.block_size)


def make_sets(rows, block_size=BLOCK_SIZE, seed=42):
    """rows: dicts with trial_id, cluster, country, ai_correct_top1 -> list of sets of trial_ids."""
    rng = random.Random(seed)
    keyed = [(bool(r["ai_correct_top1"]), r["cluster"], r["country"], rng.random(), r["trial_id"])
             for r in rows]
    order = [k[-1] for k in sorted(keyed)]
    n_sets = len(order) // block_size
    sets = [order[j:n_sets * block_size:n_sets] for j in range(n_sets)]
    rng.shuffle(sets)  # set numbers carry no information about AI accuracy
    return sets


def cycle_length(n_sets):
    return 2 * n_sets


def assign(p, n_sets):
    """Participant p (0-based) -> [(set_index, condition), (set_index, condition)] for Block A, B."""
    cycle, slot = divmod(p, cycle_length(n_sets))
    k = slot // 2
    d = 1 + cycle % (n_sets - 1)
    first, second = k, (k + d) % n_sets
    if p % 2 == 0:
        return [(first, HUMAN), (second, HUMAN_AI)]
    return [(first, HUMAN_AI), (second, HUMAN)]


def choose_practice(practice_ids, seed=42):
    """Two distinct trials (outside the study set) -> {condition: trial_id}."""
    a, b = random.Random(seed).sample(sorted(practice_ids), 2)
    return {HUMAN: a, HUMAN_AI: b}


def build_sequence(p, sets, practice, seed=42):
    """Full ordered trial list for participant p: practice trials, then Block A, then Block B.

    Trial order inside a block is shuffled per participant (seeded).
    """
    blocks = assign(p, len(sets))
    rng = random.Random(seed * 100003 + p)
    seq = [{"block": "practice", "condition": cond, "trial_id": practice[cond], "is_practice": True}
           for _, cond in blocks if practice]
    for name, (s, cond) in zip("AB", blocks):
        ids = list(sets[s])
        rng.shuffle(ids)
        seq += [{"block": name, "condition": cond, "trial_id": t, "is_practice": False} for t in ids]
    for i, trial in enumerate(seq):
        trial["trial_order"] = i
    return seq
