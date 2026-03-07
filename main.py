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
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="📝 Text Analyzer",
    description="""
Summarizes text, extracts **action items** and **key decisions**.

| Endpoint | Use |
|---|---|
| `POST /analyze/text` | Paste text directly |
| `POST /analyze/file` | Upload a text file |
""",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        description="The text you want to analyze",
        json_schema_extra={
            "example": (
                "The team reviewed Q3 results. Revenue dropped 12%. "
                "The CEO decided to cut marketing spend by 20%. "
                "John will prepare a revised forecast by Friday. "
                "Sarah will lead the cost-reduction task force."
            )
        },
    )

class AnalysisResult(BaseModel):
    summary: str
    actions: list[str]
    decisions: list[str]

# ---------------------------------------------------------------------------
# Core — single function, does everything
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
                "messages": [{
                    "role": "user",
                    "content": f"""Analyze the text below. Return ONLY valid JSON with these keys:
- "summary": 2-5 sentence summary
- "actions": list of actionable tasks
- "decisions": list of key decisions or conclusions

TEXT:
{text}"""
                }]
            },
            timeout=45,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Cannot reach LLM. Check OPENROUTER_URL.")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="LLM request timed out.")
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    try:
        raw = response.json()["choices"][0]["message"]["content"]
        clean = raw.strip().strip("```json").strip("```").strip()
        data = json.loads(clean)
        return AnalysisResult(
            summary=data.get("summary", ""),
            actions=data.get("actions", []),
            decisions=data.get("decisions", []),
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
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        raise HTTPException(status_code=422, detail="Could not decode file.")
    return analyze(text)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)