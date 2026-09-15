"""Shared definitions for Number Knot (art/ml-art-3d.md §6): models, templates, token checks.

All templates are declared here and nowhere else. Token-level construction:
every prompt is `prefix_ids + [target_id]`, and the hidden state is read at the target token
(the last position). The prefix is tokenised once and asserted identical for every target,
so a number and a random token go through exactly the same id-level pipeline.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"

OLMO = "allenai/OLMo-2-0425-1B"
QWEN = "Qwen/Qwen3-0.6B"

# ---- OLMo-2 numbers ---------------------------------------------------------------------------
# "{a}+" from the plan is dropped: in a causal model the state at the number token cannot see the
# "+" that follows, so it is identical to "{a}". The K&T GPT-J prefix replaces it.
NUMBER_TEMPLATES = [
    "{a}",                          # directly after BOS
    "The number {a}",
    "x = {a}",
    "Output ONLY a number. {a}",    # Kantamneni & Tegmark's GPT-J prompt prefix
]
N_NUMBERS = 1000
N_RANDOM = 1000

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]

# ---- Qwen3 days: target is " <Day>" with its leading space, so the prefix ends without a space.
DAY_TEMPLATES = [
    "Today is{w}", "Tomorrow is{w}", "Yesterday was{w}", "The meeting is on{w}",
    "I will see you on{w}", "The store is closed on{w}", "My favourite day of the week is{w}",
    "We go swimming every{w}", "The package arrived last{w}", "The deadline is next{w}",
    "She was born on a{w}", "Let's have lunch on{w}", "The concert takes place this{w}",
    "Every week, the garbage is collected on{w}", "It happened on a rainy{w}",
    "The office opens again on{w}", "Q: What day is it? A:{w}", "Schedule:{w}",
    "He always calls his mother on{w}", "The report was due on{w}",
    "Classes resume on{w}", "Day of the week:{w}", "The flight leaves early on{w}",
    "Our team plays football each{w}",
]

MONTH_TEMPLATES = [
    "The month is{w}", "She was born in{w}", "The festival is held every{w}",
    "The school year starts in{w}", "It snowed heavily last{w}", "The contract expires in{w}",
    "We are moving house in{w}", "My favourite month is{w}", "The report was published in{w}",
    "Harvest usually begins in{w}", "The conference takes place in early{w}",
    "They got married in{w}", "Month:{w}", "Q: What month is it? A:{w}",
    "The shop reopens next{w}", "Prices rose sharply in{w}", "The project started last{w}",
    "Rent is due at the end of{w}", "The election will be held in{w}",
    "The garden is most beautiful in{w}", "Tickets go on sale in{w}",
    "The baby is due in late{w}", "Sales were weak during{w}", "The survey closed in mid{w}",
]
assert len(DAY_TEMPLATES) >= 20 and len(MONTH_TEMPLATES) >= 20


def number_prompt_ids(tok, template: str, bos_id: int, a: int) -> tuple[list[int], list[int], int]:
    """Return (prefix_ids, full_ids, target_id) for integer a; asserts the number is one token."""
    text = template.format(a=a)
    ids = [bos_id] + tok(text, add_special_tokens=False)["input_ids"]
    single = tok(str(a), add_special_tokens=False)["input_ids"]
    assert len(single) == 1, (a, single)
    assert ids[-1] == single[0], (template, a, ids, single)
    assert tok.decode([ids[-1]]) == str(a)
    return ids[:-1], ids, ids[-1]


def build_number_prompts(tok, bos_id: int, n: int = N_NUMBERS):
    """prefix per template (asserted identical over a) and the 1000 number token ids."""
    prefixes, num_ids = [], None
    for t in NUMBER_TEMPLATES:
        pref0 = None
        ids_t = []
        for a in range(n):
            pref, _, tid = number_prompt_ids(tok, t, bos_id, a)
            if pref0 is None:
                pref0 = pref
            assert pref == pref0, (t, a, pref, pref0)
            ids_t.append(tid)
        if num_ids is None:
            num_ids = ids_t
        assert ids_t == num_ids
        prefixes.append(pref0)
    return prefixes, num_ids


_ALPHA = re.compile(r"^[A-Za-z]{2,}$")


def random_token_ids(tok, n: int, seed: int = 0, exclude: set[int] | None = None) -> list[int]:
    """n random vocabulary tokens that decode to 2+ ASCII letters with no leading space
    (numbers carry no leading space either) and survive a decode->encode round trip as one token."""
    import numpy as np
    exclude = exclude or set()
    special = set(tok.all_special_ids)
    cand = []
    for i in range(len(tok)):
        if i in special or i in exclude:
            continue
        s = tok.decode([i])
        if _ALPHA.match(s) and tok(s, add_special_tokens=False)["input_ids"] == [i]:
            cand.append(i)
    rng = np.random.default_rng(seed)
    # random order: label i <-> token ids[i]; never sorted (id order tracks BPE merge order)
    return rng.choice(cand, size=n, replace=False).tolist(), len(cand)


def word_prompts(tok, templates: list[str], words: list[str]):
    """Return prefix ids per template and the target id per word (' <word>' must be one token),
    asserting that tokenising the full string gives prefix + [target]."""
    targets = []
    for w in words:
        ids = tok(" " + w, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, (w, ids)
        targets.append(ids[0])
    prefixes = []
    for t in templates:
        pref = tok(t.format(w=""), add_special_tokens=False)["input_ids"]
        for w, tid in zip(words, targets):
            full = tok(t.format(w=" " + w), add_special_tokens=False)["input_ids"]
            assert full == pref + [tid], (t, w, full, pref, tid)
        prefixes.append(pref)
    return prefixes, targets
