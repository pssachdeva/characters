import re


NUMBERED_LINE_RE = re.compile(r"^\s*\d+[\.\):-]?\s+(.*\S)\s*$")
PREFIX_RE = re.compile(r"^\s*(?:\d+[\.\):-]?\s+|[-*]\s+)")


def parse_generated_questions(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    numbered = [match.group(1).strip() for line in lines if (match := NUMBERED_LINE_RE.match(line))]
    if numbered:
        return [_clean_question(line) for line in numbered if _clean_question(line)]
    return [_clean_question(PREFIX_RE.sub("", line).strip()) for line in lines if _clean_question(PREFIX_RE.sub("", line).strip())]


def _clean_question(text: str) -> str:
    return text.strip().strip('"').strip("'").strip()
