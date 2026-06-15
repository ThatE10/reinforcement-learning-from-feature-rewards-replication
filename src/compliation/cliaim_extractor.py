"""Stage 2: extract factual claims from each student response using Gemini.

Run standalone:
    python src/compliation/cliaim_extractor.py

Or call run(db, cfg) from pipeline.py.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError, model_validator
from tqdm import tqdm
import yaml

if TYPE_CHECKING:
    from src.data.writers import CompilationDB


# ── Prompt ────────────────────────────────────────────────────────────────────

def EXTRACTION_PROMPT(instruction: str, response: str) -> str:
    return f"""You are an entity extraction specialist. Your task is to identify ALL factual claims and entities in a model's completion that could potentially be verified.

## Input
**Instruction:**
<instruction>{instruction}</instruction>
**Completion to analyze:**
<completion>{response}</completion>

## Your Task
Extract ALL falsifiable entities and claims from the completion. DO NOT verify them - just identify and extract them. Your goal is to maximize recall (find everything) rather than precision.

## Entity Types to Extract
Extract ALL instances of (not exclusively):
- **People**: Any names, titles, roles, affiliations mentioned
- **Organizations**: Companies, institutions, agencies, groups
- **Locations**: Cities, countries, addresses, venues, geographic features
- **Dates/Times**: Specific dates, years, time periods, ages, durations
- **Events**: Meetings, conferences, historical events, incidents
- **Numbers**: Statistics, percentages, amounts, measurements, rankings
- **Citations**: Papers, books, authors, publications, studies
- **Technical Terms**: Formulas, specifications, scientific concepts, definitions
- **Products/Services**: Software, tools, platforms, models
- **Links**: URLs, websites, email addresses
- **Specific Claims**: Any factual assertion that could be verified

## Extraction Guidelines
1. **MAXIMIZE COVERAGE**: When in doubt, extract it. Better to have too many entities than miss important ones.
2. **EXACT TEXT MATCHING**: Copy entities EXACTLY as they appear, including ALL formatting (*, **, _, `, etc.)
3. **UNAMBIGUOUS SPANS**: Extract the SMALLEST meaningful unit that is UNIQUELY IDENTIFIABLE in the completion:
   - Extract the minimal substring that is unique and identifiable in the completion
   - If needed, extract a larger substring to ensure unique identification (e.g., "MIT researchers" instead of just "MIT" if "MIT" appears multiple times)
   - Extract "**Harvard University**" if that's how it appears with bold
   - The entity text should be specific enough that we can locate EXACTLY which occurrence in the completion you mean
4. **GRANULAR EXTRACTION**: Break compound claims into individual entities
5. **INCLUDE AMBIGUOUS CASES**: If something might be verifiable, include it (but ensure it's still uniquely identifiable in the text)
6. **AVOID DUPLICATES**: Do not extract the same entity text multiple times - each unique substring should appear only once in your output
7. **EXTRACT ALL**: Do not exclude anything that could potentially be verifiable

## Output Format
Return ONLY a JSON array of objects ordered by appearance in text — no markdown fences, no extra keys:
[
  {{
    "text": "The exact substring from completion (with all formatting)",
    "context_hint": "Brief note about what this entity refers to (helps with later verification)"
  }}
]

## Example
For the completion: "OpenAI released GPT-4 in March 2023, achieving 86.4% on the MMLU benchmark."

[
  {{"text": "OpenAI released GPT-4 in March 2023", "context_hint": "Release claim with date"}},
  {{"text": "achieving 86.4% on the MMLU benchmark", "context_hint": "Performance claim"}}
]

Remember: Extract EVERYTHING that could potentially be fact-checked. Do NOT verify or judge accuracy.
"""


# ── Response schema ───────────────────────────────────────────────────────────
# Gemini returns a raw JSON array; model_validator wraps it into ClaimList.

class Claim(BaseModel):
    text: str = Field(description="Exact substring from the completion")
    context_hint: str = Field(description="Brief note about what this entity refers to")


class ClaimList(BaseModel):
    claims: list[Claim]

    @model_validator(mode="before")
    @classmethod
    def wrap_array(cls, v: object) -> object:
        """Allow model_validate([...]) in addition to model_validate({"claims": [...]})."""
        if isinstance(v, list):
            return {"claims": v}
        return v


# ── Gemini call ───────────────────────────────────────────────────────────────

def _call_gemini(
    question: str,
    student_response: str,
    client: genai.Client,
    model_name: str,
) -> genai.types.GenerateContentResponse:
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=EXTRACTION_PROMPT(question, student_response))],
        )
    ]
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
    )
    return client.models.generate_content(model=model_name, contents=contents, config=config)


def _extract_with_retry(
    question: str,
    student_response: str,
    client: genai.Client,
    model_name: str,
    max_retries: int = 3,
) -> ClaimList | None:
    """Call Gemini and validate the response; retry on parse errors or rate limits.

    Returns None if all attempts fail.
    """
    from src.data.schemas import validate_gemini_response

    for attempt in range(max_retries):
        try:
            resp = _call_gemini(question, student_response, client, model_name)
            return validate_gemini_response(resp.text, ClaimList)

        except (ValueError, ValidationError) as e:
            if attempt == max_retries - 1:
                tqdm.write(f"[warn] claim parse failed after {max_retries} attempts: {e}")
                return None
            time.sleep(2 ** attempt)

        except Exception as e:
            err = str(e)
            if any(tok in err for tok in ("429", "503", "RESOURCE_EXHAUSTED", "overloaded")):
                wait = 5 * (2 ** attempt)
                tqdm.write(f"[warn] rate limited — retrying in {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
            else:
                tqdm.write(f"[error] unexpected API error: {e}")
                return None

    return None


# ── Entry point ───────────────────────────────────────────────────────────────

_CHUNK_SIZE = 200


def run(db: "CompilationDB", cfg: dict) -> None:
    """Process all DB records that don't yet have extracted_entities, writing
    extracted_entities (list[str] of claim text spans) and entity_context
    (list[str] of context hints) back to each row.
    """
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set — add it to .env or the environment")

    client = genai.Client(api_key=api_key)
    model_name = cfg.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
    max_retries = cfg.get("CLAIM_EXTRACTION_MAX_RETRIES", 3)

    total_updated = 0
    total_skipped = 0

    while True:
        records = db.fetch_unprocessed("extracted_entities", limit=_CHUNK_SIZE)
        if not records:
            break

        for record in tqdm(records, desc="Extracting claims", unit="rec"):
            question = record.get("question")
            student_response = record.get("student_response")

            if not question or not student_response:
                total_skipped += 1
                continue

            result = _extract_with_retry(
                question,
                student_response,
                client,
                model_name,
                max_retries=max_retries,
            )

            if result is None:
                total_skipped += 1
                continue

            db.update_record(record["id"], {
                "extracted_entities": [c.text for c in result.claims],
                "entity_context":     [c.context_hint for c in result.claims],
            })
            total_updated += 1

    print(
        f"\n[claim_extractor] Done — "
        f"{total_updated} records updated, {total_skipped} skipped"
    )


# ── Standalone ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.data.writers import CompilationDB  # noqa: E402 (needed after sys.path insert)

    with open("configs/base.yaml") as f:
        cfg = yaml.safe_load(f)

    db_path = cfg.get("DB_PATH", "outputs/compilation.db")
    with CompilationDB(db_path) as db:
        run(db, cfg)
