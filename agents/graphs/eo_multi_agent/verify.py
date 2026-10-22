"""Rule-based checks on the final answer before it reaches the user.

Two failure modes of small models, both seen live in EVE:

- They rewrite the ``/artifacts/<id>`` image links of tool results into URLs
  that do not exist. The backend already appends a correct stub for every
  image the answer does not reference, so the safe fix is to drop the image
  markdown from the answer instead of repairing it.
- They write numbers no tool returned. :func:`unsupported_numbers` lists every
  number in the answer that cannot be traced back to the evidence (the user's
  question, the tool results and the specialists' reports), so the reporter
  can be asked to fix exactly those. The arguments the specialists passed to
  the tools count as evidence too (a chart of "36 months" is a choice, not a
  claim).
"""

from __future__ import annotations

import re
from typing import Iterable, List, Set

# A number with optional sign and digit groups: 18,244 · 18.244 · -2.1 · 0,49
_NUMBER = re.compile(r"(?<![\w.,])[-−]?\d+(?:[.,  ]\d+)*")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]*)[^)]*\)")
_ARTIFACT_URL = re.compile(r"/artifacts/[\w-]+")

# Small counts ("5 farms", "3 months") and years are not measurements; checking
# them would only trigger rewrites for numbers the user would never doubt.
SMALL_COUNT = 12
YEARS = range(1900, 2101)

# A rounded value still counts as supported ("18,300 ha" for 18,271 ha).
RELATIVE_TOLERANCE = 0.01
ABSOLUTE_TOLERANCE = 0.051


def _readings(token: str) -> Set[float]:
    """Every way to read *token*: comma or dot as decimal or thousands mark."""
    token = token.replace("−", "-").replace(" ", ",").replace(" ", ",")
    readings = set()
    for thousands, decimal in ((",", "."), (".", ",")):
        groups = token.split(decimal)
        if len(groups) > 2:
            continue
        whole = groups[0].replace(thousands, "")
        text = whole if len(groups) == 1 else f"{whole}.{groups[1]}"
        if thousands in groups[0] and not all(
            len(g) == 3 for g in groups[0].lstrip("-").split(thousands)[1:]
        ):
            continue
        try:
            readings.add(float(text))
        except ValueError:
            continue
    return readings


def numbers(text: str) -> List[str]:
    """The number tokens in *text*, in order."""
    return _NUMBER.findall(text or "")


def _is_exempt(token: str) -> bool:
    readings = _readings(token)
    return bool(readings) and all(
        (r.is_integer() and abs(r) <= SMALL_COUNT) or (r.is_integer() and int(r) in YEARS)
        for r in readings
    )


def _close(value: float, known: Iterable[float]) -> bool:
    # Magnitudes only: answers write -7.74 as "7.74° W" and -2.1 as "2.1 below normal".
    value = abs(value)
    for k in (abs(k) for k in known):
        # A fraction in a result may be a percentage in the answer (0.49 -> 49 %).
        for candidate in (k, k * 100) if 0 < abs(k) < 1 else (k,):
            if abs(value - candidate) <= max(
                ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE * abs(candidate)
            ):
                return True
    return False


def unsupported_numbers(answer: str, evidence: str) -> List[str]:
    """Numbers in *answer* that no reading of *evidence* supports, deduplicated."""
    known: Set[float] = set()
    for token in numbers(evidence):
        known.update(_readings(token))
    missing: List[str] = []
    for token in numbers(answer):
        if token in missing or _is_exempt(token):
            continue
        readings = _readings(token)
        if readings and not any(_close(r, known) for r in readings):
            missing.append(token)
    return missing


def clean_links(answer: str, evidence: str) -> str:
    """Drop image markdown and links to artifacts the tools never produced.

    Images are attached by the backend after the answer, so an image line here
    is either a duplicate or a broken rewrite. A text link to a real artifact
    is kept; any other ``/artifacts/`` or ``resource://`` link keeps its text
    and loses the URL.
    """
    produced = set(_ARTIFACT_URL.findall(evidence or ""))
    text = _IMAGE.sub("", answer or "")

    def relink(match: re.Match) -> str:
        label, url = match.group(1), match.group(2)
        if url.startswith("resource://") or (
            "/artifacts/" in url and url not in produced
        ):
            return label
        return match.group(0)

    text = _LINK.sub(relink, text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
