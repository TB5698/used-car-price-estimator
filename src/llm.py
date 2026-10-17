"""Talk to the LLM through any OpenAI-compatible API (Groq by default).

The LLM has two small jobs: turn a plain-English message into car fields, and
turn numbers computed in Python into a short explanation. It never decides
whether a car can be priced; src/validate.py does that.

Settings come from .env (see .env.example): LLM_API_KEY, LLM_MODEL and, to use a
provider other than Groq, LLM_BASE_URL. To see which models your key can use, run
from the repo root:
    python -m src.llm
"""
import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from src.validate import MODEL_COLUMNS

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

EXTRACT_PROMPT = """You read a message about a used car and fill in a JSON object.
Use exactly these keys: year, manufacturer, model, odometer, type, drive, fuel, \
transmission, condition, title_status.

Rules:
- Use null for anything the message does not say. Never guess or infer a value \
that was not stated.
- year and odometer are plain numbers (90k miles is 90000).
- manufacturer is the make (toyota, ford). model is the model name only (camry, f-150).
- type is the body style: sedan, suv, pickup, coupe, hatchback, wagon, van, convertible.
- drive is fwd, rwd or 4wd. fuel is gas, diesel, hybrid or electric. \
transmission is automatic or manual.
- condition is new, like new, excellent, good, fair or salvage. \
title_status is clean, rebuilt, salvage or lien.
- If the message is not about a used car, set every key to null.

Reply with the JSON object only."""

EXPLAIN_PROMPT = """You explain a used car price estimate to a shopper in plain, \
friendly language.
Use only the facts you are given. Do not add numbers, reasons or advice that are \
not in them.
Write every number as digits with commas, like 12,300 (never in words like \
"twelve thousand" and never 12.3k), and do not calculate new numbers.
Write 3 to 5 sentences: the estimated price and range first, then what moves the \
price, then any caveats."""


def get_client():
    """OpenAI client for the provider in .env. The key comes from .env or the environment."""
    load_dotenv()
    key = os.environ.get("LLM_API_KEY")
    if not key:
        raise RuntimeError(
            "LLM_API_KEY is not set. Add it to a .env file in the repo root "
            "(see .env.example)."
        )
    base_url = os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL
    return OpenAI(base_url=base_url, api_key=key)


def get_model_name():
    """Model id from .env or the environment."""
    load_dotenv()
    name = os.environ.get("LLM_MODEL")
    if not name:
        raise RuntimeError(
            "LLM_MODEL is not set. Run `python -m src.llm` to list the models "
            "your key can use, then add one to .env (see .env.example)."
        )
    return name


def parse_json_object(text):
    """First JSON object in a model reply (it may add code fences or extra words)."""
    if not text:
        return {}
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def extract_car(message, client, model):
    """Car fields the user stated in plain English. Anything not said is None."""
    reply = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user", "content": message},
        ],
    )
    data = parse_json_object(reply.choices[0].message.content)
    return {name: data.get(name) for name in MODEL_COLUMNS}


def explain_estimate(facts, client, model):
    """Short plain-language explanation of numbers that Python already computed."""
    reply = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=[
            {"role": "system", "content": EXPLAIN_PROMPT},
            {"role": "user", "content": json.dumps(facts, indent=2)},
        ],
    )
    return reply.choices[0].message.content.strip()


if __name__ == "__main__":
    for item in get_client().models.list():
        print(item.id)
