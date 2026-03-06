import os
import sys
import json
import argparse
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL")
API_URL = os.getenv("OPENROUTER_URL")


def read_file_dynamic(path):
    """
    Attempt to read any file containing text using multiple encodings.
    """
    encodings = ["utf-8", "latin-1", "utf-16"]

    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue

    raise ValueError(f"Could not read file as text: {path}")


def read_input(input_source=None):
    """
    Read input dynamically from:
    - Any file containing text
    - stdin
    - direct text input
    """

    if input_source and os.path.exists(input_source):
        return read_file_dynamic(input_source)

    if not sys.stdin.isatty():
        return sys.stdin.read()

    if input_source:
        return input_source

    raise ValueError("No input provided.")


def call_llm(text):
    """
    Call OpenRouter Nemotron model
    """

    prompt = f"""
Extract structured information from the following text.

Return ONLY valid JSON with keys:
summary: concise summary (2-5 sentences)
actions: list of action items
decisions: list of decisions

Rules:
- Actions must be actionable tasks
- Decisions must represent conclusions or choices
- Deduplicate similar items
- Each item must be short (1 line)

TEXT:
{text}
"""

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0
        }
    )

    if response.status_code == 429:
        raise Exception("Rate limit exceeded. Wait a few minutes and try again.")
    
    response.raise_for_status()

    data = response.json()

    return data["choices"][0]["message"]["content"]


def parse_response(text):
    """
    Safely parse model response
    """

    try:
        data = json.loads(text)

        actions = list(set(data.get("actions", [])))
        decisions = list(set(data.get("decisions", [])))

        return {
            "summary": data.get("summary", ""),
            "actions": actions,
            "decisions": decisions
        }

    except Exception:
        return {
            "summary": "",
            "actions": [],
            "decisions": []
        }


def main():

    parser = argparse.ArgumentParser(
        description="Dynamic Text Processor"
    )

    parser.add_argument(
        "--input",
        help="File path OR raw text input"
    )

    args = parser.parse_args()

    try:
        text = read_input(args.input)
        response = call_llm(text)
        result = parse_response(response)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()