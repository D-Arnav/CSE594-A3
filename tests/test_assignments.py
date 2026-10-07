import random
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "study_app"))
from assignments import (HUMAN, HUMAN_AI, assign, build_sequence, choose_practice,  # noqa: E402
                         cycle_length, make_sets)
from common import CLUSTER_OF, load_config, countries  # noqa: E402


def study_rows(seed=0):
    rng = random.Random(seed)
    rows = []
    labels = countries(load_config())
    for i, c in enumerate(labels * (200 // len(labels))):  # the real study size
        rows.append({"trial_id": f"S{i + 1:03d}", "country": c, "cluster": CLUSTER_OF[c],
                     "ai_correct_top1": rng.random() < 0.55})
    return rows


@pytest.mark.parametrize("size", [8, 6])
def test_make_sets_partition_and_stratification(size):
    rows = study_rows()
    sets = make_sets(rows, size, seed=42)
    n = len(rows) // size
    assert len(sets) == n and all(len(s) == size for s in sets)
    flat = [t for s in sets for t in s]
    assert len(set(flat)) == len(flat) == n * size  # disjoint; 200 % 6 = 2 left over
    by_id = {r["trial_id"]: r for r in rows}
    correct = [sum(by_id[t]["ai_correct_top1"] for t in s) for s in sets]
    assert max(correct) - min(correct) <= 1
    for s in sets:
        n_countries = len({r["country"] for r in rows})
        per_country = Counter(by_id[t]["country"] for t in s)
        assert max(per_country.values()) <= -(-size // n_countries) + 1  # spread, not lumped
        assert len(per_country) >= min(n_countries, size // 2)
    assert make_sets(rows, size, seed=42) == sets  # deterministic


@pytest.mark.parametrize("n_sets", [25, 33])
def test_rotation_balanced_over_every_cycle(n_sets):
    L = cycle_length(n_sets)
    for cycle in range(n_sets):  # every pairing offset d = 1..n_sets-1 (and wrap-around)
        slots = Counter()
        cond = Counter()
        for p in range(cycle * L, (cycle + 1) * L):
            blocks = assign(p, n_sets)
            assert blocks[0][0] != blocks[1][0]  # no participant sees an image twice
            assert {c for _, c in blocks} == {HUMAN, HUMAN_AI}
            assert blocks[0][1] == (HUMAN if p % 2 == 0 else HUMAN_AI)  # order alternates by parity
            for pos, (s, c) in enumerate(blocks):
                slots[s, pos, c] += 1
                cond[s, c] += 1
        # each set in each (position, condition) slot exactly once per cycle
        assert set(slots.values()) == {1} and len(slots) == 4 * n_sets
        assert set(cond.values()) == {2} and len(cond) == 2 * n_sets


def test_conditions_balanced_after_every_pair():
    n_sets = 25
    cond = Counter()
    for p in range(4 * cycle_length(n_sets)):
        for s, c in assign(p, n_sets):
            cond[s, c] += 1
        if p % 2 == 1:
            assert all(cond[s, HUMAN] == cond[s, HUMAN_AI] for s in range(n_sets))


def test_build_sequence():
    sets = make_sets(study_rows())
    practice = choose_practice([f"D{i:03d}" for i in range(1, 201)])
    assert practice[HUMAN] != practice[HUMAN_AI]
    for p in range(cycle_length(len(sets))):
        seq = build_sequence(p, sets, practice)
        assert [t["block"] for t in seq] == ["practice"] * 2 + ["A"] * 8 + ["B"] * 8
        assert [t["trial_order"] for t in seq] == list(range(18))
        assert [t["condition"] for t in seq[:2]] == [seq[2]["condition"], seq[-1]["condition"]]
        assert all(t["is_practice"] == (t["block"] == "practice") for t in seq)
        ids = [t["trial_id"] for t in seq[2:]]
        assert len(set(ids)) == 16 and not set(ids) & set(practice.values())
        (sa, _), (sb, _) = assign(p, len(sets))
        assert set(ids[:8]) == set(sets[sa]) and set(ids[8:]) == set(sets[sb])
    assert build_sequence(3, sets, practice) == build_sequence(3, sets, practice)
