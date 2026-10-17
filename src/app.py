"""One conversation with the price estimator. The Streamlit page will call Chat.send().

Each turn:
    1. The LLM pulls any car details out of the message (null for anything not said).
    2. Python merges them into the car so far, so answers across turns add up.
    3. Python validates the car (src/validate.py). Missing or unusable details get a
       clarifying question, cars outside the training data get a clear "out of scope".
    4. A complete car is priced by the model (src/estimate.py).
    5. The LLM explains the numbers. Python checks that it added no numbers of its own
       and falls back to a plain summary if it did, or if the LLM call fails.

The LLM never decides whether a car can be priced, and never produces a price.
"""
import json
import re
from dataclasses import dataclass

from src.estimate import PriceEstimator, build_facts
from src.llm import explain_estimate, extract_car, get_client, get_model_name
from src.preprocess import load_config
from src.validate import ALIASES, BLANKS, question_for, validate_car

OFF_TOPIC_REPLY = (
    "I estimate the asking price of used cars. Tell me about one, for example: "
    '"2016 Toyota Camry, 90,000 miles, sedan, front wheel drive, gas, automatic".'
)
NO_DETAILS_REPLY = "I did not catch any car details in that."
LLM_DOWN_REPLY = "I could not reach the language model just now. Please try again in a moment."


@dataclass
class Reply:
    kind: str            # "estimate", "question", "out_of_scope", "off_topic" or "error"
    text: str            # what to show the user
    headline: str = None   # the price line, always written by Python
    estimate: object = None
    facts: dict = None


def dollars(value):
    return f"${value:,.0f}"


def _stated(extracted):
    """Only the fields the user actually gave."""
    return {name: value for name, value in extracted.items()
            if value is not None and str(value).strip().lower() not in BLANKS}


def _same_text(name, a, b):
    a, b = str(a).strip().lower(), str(b).strip().lower()
    aliases = ALIASES.get(name, {})
    return aliases.get(a, a) == aliases.get(b, b)


def _numbers(text):
    """Every number in the text as a float (12,300 and 12300 are the same number)."""
    found = re.findall(r"\d[\d,]*(?:\.\d+)?", str(text))
    return {float(item.replace(",", "").rstrip(".")) for item in found if item.strip(",")}


NUMBER_WORDS = re.compile(r"\b(hundred|thousand|million)\b", re.IGNORECASE)


def adds_no_new_numbers(text, facts):
    """True if every number of 10 or more in the text appears in the facts.

    Numbers written as words ("twelve thousand") cannot be checked, so they fail too.
    """
    if NUMBER_WORDS.search(text):
        return False
    allowed = _numbers(json.dumps(facts))
    return all(number in allowed for number in _numbers(text) if number >= 10)


def headline_for(estimate):
    return (f"Estimated asking price: {dollars(estimate.price)} "
            f"(likely range {dollars(estimate.low)} to {dollars(estimate.high)})")


def plain_summary(facts):
    """The explanation written by Python, used when the LLM's version cannot be trusted."""
    lines = list(facts["assumptions"])
    for driver in facts["what_moves_the_price"]:
        change = driver["price_change"]
        direction = "higher" if change > 0 else "lower"
        lines.append(f"With {driver['if_instead']}: about {dollars(driver['new_estimate'])} "
                     f"({dollars(abs(change))} {direction}).")
    lines.extend(facts["caveats"])
    lines.append("This is a typical asking price from 2021 listings, not a guaranteed sale price.")
    return " ".join(lines)


class Chat:
    def __init__(self, client, model_name, estimator, cfg):
        self.client = client
        self.model_name = model_name
        self.estimator = estimator
        self.cfg = cfg          # data section of config.yaml
        self.fields = {}        # car details gathered so far

    @classmethod
    def from_env(cls):
        """Real client, model and estimator. Needs .env and the files in models/."""
        return cls(get_client(), get_model_name(), PriceEstimator.load(), load_config())

    def reset(self):
        self.fields = {}

    def send(self, message):
        """Handle one user message and return a Reply."""
        if not message or not message.strip():
            return Reply("off_topic", OFF_TOPIC_REPLY)

        try:
            stated = _stated(extract_car(message, self.client, self.model_name))
        except Exception:
            return Reply("error", LLM_DOWN_REPLY)

        if not stated:
            # nothing usable in the message: off topic, unless a car is still being filled in
            if validate_car(self.fields, self.cfg).status == "ok":
                return Reply("off_topic", OFF_TOPIC_REPLY)
            if not self.fields:
                return Reply("off_topic", OFF_TOPIC_REPLY)
            return self._answer(NO_DETAILS_REPLY)

        if self._is_different_car(stated):
            self.fields = {}
        self.fields.update(stated)
        return self._answer()

    def _is_different_car(self, stated):
        """A new make or model means the user moved on to another car."""
        for name in ("manufacturer", "model"):
            old, new = self.fields.get(name), stated.get(name)
            if old is not None and new is not None and not _same_text(name, old, new):
                return True
        return False

    def _answer(self, prefix=""):
        """Price the car if it is complete, otherwise ask for what is missing."""
        result = validate_car(self.fields, self.cfg)
        if result.status == "ok":
            return self._price(result)
        kind = "out_of_scope" if result.status == "out_of_scope" else "question"
        return Reply(kind, " ".join(part for part in [prefix, question_for(result)] if part))

    def _price(self, result):
        estimate = self.estimator.estimate(result.car)
        facts = build_facts(result.car, estimate, result.assumptions, result.warnings)
        try:
            text = explain_estimate(facts, self.client, self.model_name)
            if not text or not adds_no_new_numbers(text, facts):
                text = plain_summary(facts)
        except Exception:
            text = plain_summary(facts)
        return Reply("estimate", text, headline_for(estimate), estimate, facts)


def main():
    """Try the whole pipeline in the terminal: python -m src.app"""
    chat = Chat.from_env()
    print("Describe a used car (blank line to quit).")
    while True:
        message = input("\nyou> ").strip()
        if not message:
            break
        reply = chat.send(message)
        if reply.headline:
            print(f"\n{reply.headline}")
        print(reply.text)


if __name__ == "__main__":
    main()
