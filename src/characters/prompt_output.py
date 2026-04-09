import re


NUMBERED_LINE_RE = re.compile(r"^\s*\d+[\.\):-]?\s+(.*\S)\s*$")
PREFIX_RE = re.compile(r"^\s*(?:\d+[\.\):-]?\s+|[-*]\s+)")
HEADING_RE = re.compile(r"^(?:trait|questions|seed questions|additional questions)\s*:?.*$", re.IGNORECASE)


def parse_generated_questions(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    numbered = [match.group(1).strip() for line in lines if (match := NUMBERED_LINE_RE.match(line))]
    candidates = numbered or [PREFIX_RE.sub("", line).strip() for line in lines]
    parsed: list[str] = []
    for candidate in candidates:
        cleaned = _clean_question(candidate)
        if cleaned and not HEADING_RE.match(cleaned):
            parsed.append(cleaned)
    return parsed


def _clean_question(text: str) -> str:
    return text.strip().strip('"').strip("'").strip()
