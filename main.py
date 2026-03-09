"""
Text Analyzer API
Run:  uvicorn main:app --reload
Docs: http://localhost:8000/docs
"""

import os
import json
import requests
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from pydantic import BaseModel

load_dotenv()

# Read timeout from .env so it can be tuned per environment without touching code.
# e.g. LLM_TIMEOUT=60 for large models, LLM_TIMEOUT=15 for fast/small ones.
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="📝 Text Analyzer",
    description="""

""",
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AnalysisResult(BaseModel):
    summary: str
    action_items: list[str]
    risks: list[str]
    priority_tasks: list[str]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PROMPT = """Analyze the TEXT below and return ONLY a valid JSON object with exactly these four keys:

{{
  "summary": "",
  "action_items": [],
  "risks": [],
  "priority_tasks": []
}}

RULES FOR EACH FIELD

summary
  - 2-5 sentences describing the core topic and outcome.
  - Return "" if text is under 15 words, blank, or makes no sense.

action_items
  - Only tasks or assignments that are clearly and explicitly stated in the text.
  - Do not infer, assume, or invent anything not directly written.
  - Return [] if none found.

risks
  - Threats, blockers, or concerns that are stated or directly implied in the text.
  - Do not fabricate risks that have no basis in the text.
  - Return [] if none found.

priority_tasks
  - The most urgent items taken only from action_items — no new tasks.
  - Order by: explicit deadline first, then impact, then risk.
  - Return [] if action_items is [].

CORNER CASES
  - Text under 15 words or blank   → summary="", all three arrays []
  - No tasks in text               → action_items=[], priority_tasks=[]
  - No risks in text               → risks=[]
  - Duplicate tasks                → keep only the most specific one
  - Emotional or narrative text    → fill summary, leave arrays [] unless tasks/risks are present
  - Noisy or garbled text          → extract what is clearly readable, skip the rest
  - Non-English text               → respond in the same language as the input

BEFORE RETURNING check:
  - Valid JSON only — no markdown fences, no explanation, no extra keys
  - priority_tasks must be a subset of action_items
  - Nothing invented — every item must come directly from the text
  - Empty fields are [] or "" never null

TEXT:
{text}"""

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def analyze(text: str) -> AnalysisResult:
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty.")

    try:
        response = requests.post(
            os.environ["OPENROUTER_URL"],
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
            json={
                "model": os.environ["OPENROUTER_MODEL"],
                "temperature": 0,
                "messages": [{"role": "user", "content": PROMPT.format(text=text)}],
            },
            timeout=LLM_TIMEOUT,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Cannot reach LLM. Check OPENROUTER_URL.")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail=f"LLM did not respond within {LLM_TIMEOUT}s. Increase LLM_TIMEOUT in .env.")
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    try:
        raw = response.json()["choices"][0]["message"]["content"]
        # Isolate the outermost { ... } to handle any stray prose the model adds
        clean = raw.strip()
        start, end = clean.find("{"), clean.rfind("}")
        if start != -1 and end > start:
            clean = clean[start: end + 1]
        data = json.loads(clean)
        return AnalysisResult(
            summary=data.get("summary", ""),
            action_items=data.get("action_items", []),
            risks=data.get("risks", []),
            priority_tasks=data.get("priority_tasks", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM response: {e}")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post(
    "/analyze/plain",
    response_model=AnalysisResult,
    tags=["Analyze"],
    summary="📝 Plain text (no JSON)",
)
async def analyze_plain(text: str = Body(..., media_type="text/plain")):
    return analyze(text)


@app.post(
    "/analyze/file",
    response_model=AnalysisResult,
    tags=["Analyze"],
    summary="📁 Upload a text file",
    description="Upload any plain text file (`.txt`, `.md`, `.log`, `.csv`, etc.)",
)
async def analyze_file(file: UploadFile = File(..., description="Any plain text file")):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="File is empty.")
    return analyze(raw.decode("utf-8", errors="ignore"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)