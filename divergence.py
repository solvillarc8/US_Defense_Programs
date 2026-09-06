"""Evidence-bounded comparison of GAO, Weapons Book, and CRS retrievals."""

import openai

from config import (
    DIVERGENCE_MAX_TOKENS,
    DIVERGENCE_MODEL,
    DIVERGENCE_SOURCE_CHAR_LIMIT,
)


_SECTION_NAMES = ("AGREEMENTS", "DIVERGENCES", "DIFFERENT EMPHASIS", "SOURCE GAPS")


def _validate_sections(text: str) -> str:
    """Mechanically enforce the two most important classification invariants."""
    sections = {name: [] for name in _SECTION_NAMES}
    current = None
    for raw_line in text.splitlines():
        heading = raw_line.strip().strip("*# ").upper()
        if heading in sections:
            current = heading
            continue
        if current and raw_line.strip():
            sections[current].append(raw_line.strip())

    source_names = ("gao", "weapons book", "crs")
    agreements = [
        line for line in sections["AGREEMENTS"]
        if sum(name in line.lower() for name in source_names) >= 2
    ]
    if not agreements:
        agreements = ["- No cross-source agreement identified in the retrieved evidence."]

    silence_markers = ("does not mention", "not mentioned", "no coverage", "not cover")
    divergences = [
        line for line in sections["DIVERGENCES"]
        if not any(marker in line.lower() for marker in silence_markers)
    ]
    if not divergences:
        divergences = ["- No explicit disagreement found."]

    rebuilt = {
        "AGREEMENTS": agreements,
        "DIVERGENCES": divergences,
        "DIFFERENT EMPHASIS": sections["DIFFERENT EMPHASIS"] or ["- None identified."],
        "SOURCE GAPS": sections["SOURCE GAPS"] or ["- None identified."],
    }
    return "\n\n".join(
        f"**{name}**\n" + "\n".join(rebuilt[name]) for name in _SECTION_NAMES
    )


def analyze_divergence(program: str, source_results: dict[str, str]) -> str:
    """Classify agreements, disagreements, and source gaps inside the tool."""
    evidence = []
    for source, text in source_results.items():
        evidence.append(
            f"===== {source} =====\n{text[:DIVERGENCE_SOURCE_CHAR_LIMIT]}"
        )

    prompt = f"""Compare the retrieved evidence below for {program}.

This is an evidence-classification step inside a retrieval tool. Use only the
supplied text. Do not infer a source's attitude from silence or turn a budget
request into confidence, commitment, or actual spending.

Return exactly these sections:
AGREEMENTS
- paired facts supported by two or more sources
DIVERGENCES
- incompatible claims or materially different figures/dates/status
DIFFERENT EMPHASIS
- non-conflicting differences in what each source discusses
SOURCE GAPS
- sources with no substantive coverage of this program

Every bullet must name each source and include its exact figure or a short
supporting snippet. Cite only the program/section and page range in each Result
header. Treat dense multi-year extracted tables as ambiguous unless narrative
prose explicitly maps the figure. If no true disagreement is supported, write
"No explicit disagreement found."

Classification tests you MUST apply before writing:
- DIVERGENCE requires Source A and Source B to make affirmative, incompatible
  claims about the same property. "A reports X while B does not mention X" is
  NEVER a divergence; put that in DIFFERENT EMPHASIS or SOURCE GAPS.
- AGREEMENT requires at least two sources to support the same proposition.
  Two unrelated true statements are not an agreement.
- SOURCE GAPS must not cite an unrelated returned chunk as if it discussed the
  program. State simply that substantive coverage was not retrieved.
- Citation labels must be the header's program/section value and PDF page
  range, never "Result 1" and never a printed page label inside excerpt text.

Evidence:
{chr(10).join(evidence)}"""
    response = openai.OpenAI().chat.completions.create(
        model=DIVERGENCE_MODEL,
        max_tokens=DIVERGENCE_MAX_TOKENS,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content or ""
    return _validate_sections(raw)
