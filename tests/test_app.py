import json
from types import SimpleNamespace

from src.app import (NO_DETAILS_REPLY, OFF_TOPIC_REPLY, Chat, adds_no_new_numbers,
                     plain_summary)
from src.estimate import build_facts
from tests.test_estimate import CAR, CFG, estimator

CAR_TEXT = "2016 Toyota Camry, 90k miles, sedan, fwd, gas, automatic"
CAR_FIELDS = dict(year=2016, manufacturer="toyota", model="camry", odometer=90000,
                  type="sedan", drive="fwd", fuel="gas", transmission="automatic")


class ScriptedClient:
    """Fake LLM. Extraction replies come from a dict keyed by the user message,
    explanation replies from `explanation` (text, or an Exception to raise)."""

    def __init__(self, extractions, explanation="Placeholder."):
        self.extractions = extractions
        self.explanation = explanation
        self.explain_calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        system, user = kwargs["messages"][0]["content"], kwargs["messages"][-1]["content"]
        if "fill in a JSON object" in system:
            if isinstance(self.extractions, Exception):
                raise self.extractions
            reply = json.dumps(self.extractions.get(user, {}))
        else:
            self.explain_calls += 1
            if isinstance(self.explanation, Exception):
                raise self.explanation
            reply = self.explanation
        message = SimpleNamespace(content=reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def chat_with(extractions, explanation="Placeholder."):
    client = ScriptedClient(extractions, explanation)
    return Chat(client, "fake-model", estimator(), CFG), client


GOOD_EXPLANATION = ("Your Camry should list around 16,500 dollars, with a likely range of "
                    "13,200 to 20,600. Rebuilt title would cut it to about 11,600.")


# the three demo cases

def test_demo_normal_question_gets_an_estimate():
    chat, client = chat_with({CAR_TEXT: CAR_FIELDS}, GOOD_EXPLANATION)
    reply = chat.send(CAR_TEXT)
    assert reply.kind == "estimate"
    assert reply.headline == "Estimated asking price: $16,500 (likely range $13,200 to $20,600)"
    assert reply.text == GOOD_EXPLANATION
    assert reply.estimate.price == 16500
    assert client.explain_calls == 1


def test_demo_missing_info_gets_a_question_and_no_estimate():
    chat, client = chat_with({"2016 Camry": dict(year=2016, manufacturer="toyota", model="camry")})
    reply = chat.send("2016 Camry")
    assert reply.kind == "question"
    assert reply.estimate is None and client.explain_calls == 0
    for word in ["mileage", "body type", "drive", "fuel", "transmission"]:
        assert word in reply.text


def test_demo_off_topic_gets_the_fixed_reply():
    chat, client = chat_with({"what is the capital of France": {}})
    reply = chat.send("what is the capital of France")
    assert reply.kind == "off_topic" and reply.text == OFF_TOPIC_REPLY
    assert client.explain_calls == 0


# conversation behavior

def test_details_add_up_across_turns():
    chat, _ = chat_with({
        "2016 Camry": dict(year=2016, manufacturer="toyota", model="camry"),
        "90k miles": dict(odometer=90000),
        "sedan, fwd, gas, automatic": dict(type="sedan", drive="fwd", fuel="gas",
                                           transmission="automatic"),
    }, GOOD_EXPLANATION)
    assert chat.send("2016 Camry").kind == "question"
    second = chat.send("90k miles")
    assert second.kind == "question" and "mileage" not in second.text
    assert chat.send("sedan, fwd, gas, automatic").kind == "estimate"


def test_follow_up_changes_one_detail_and_reprices():
    chat, _ = chat_with({CAR_TEXT: CAR_FIELDS, "what if 50k miles": dict(odometer=50000)},
                        "Placeholder.")
    first = chat.send(CAR_TEXT)
    second = chat.send("what if 50k miles")
    assert second.kind == "estimate"
    assert second.estimate.price > first.estimate.price   # fewer miles, higher price
    assert chat.fields["odometer"] == 50000 and chat.fields["model"] == "camry"


def test_a_different_make_or_model_starts_a_new_car():
    chat, _ = chat_with({CAR_TEXT: CAR_FIELDS, "a 2012 Honda Civic": dict(
        year=2012, manufacturer="honda", model="civic")})
    chat.send(CAR_TEXT)
    reply = chat.send("a 2012 Honda Civic")
    assert reply.kind == "question"           # Camry details were dropped, so more is needed
    assert chat.fields == dict(year=2012, manufacturer="honda", model="civic")


def test_alias_for_the_same_make_does_not_start_a_new_car():
    chat, _ = chat_with({"a chevy": dict(manufacturer="chevy"),
                         "a Chevrolet 90k": dict(manufacturer="chevrolet", odometer=90000)})
    chat.send("a chevy")
    chat.send("a Chevrolet 90k")
    assert chat.fields["odometer"] == 90000 and chat.fields["manufacturer"] == "chevrolet"


def test_unreadable_reply_while_a_car_is_in_progress_asks_again():
    chat, _ = chat_with({"2016 Camry": dict(year=2016, manufacturer="toyota", model="camry"),
                         "hmm": {}})
    chat.send("2016 Camry")
    reply = chat.send("hmm")
    assert reply.kind == "question"
    assert reply.text.startswith(NO_DETAILS_REPLY) and "mileage" in reply.text


def test_out_of_scope_car_is_refused_clearly():
    chat, client = chat_with({"1975 Corvette": dict(year=1975, manufacturer="chevrolet",
                                                    model="corvette", odometer=50000)})
    reply = chat.send("1975 Corvette")
    assert reply.kind == "out_of_scope"
    assert "1990" in reply.text and client.explain_calls == 0


def test_blank_message_does_not_call_the_llm():
    chat, client = chat_with({})
    assert chat.send("   ").kind == "off_topic"
    assert client.explain_calls == 0


def test_reset_forgets_the_car():
    chat, _ = chat_with({CAR_TEXT: CAR_FIELDS})
    chat.send(CAR_TEXT)
    chat.reset()
    assert chat.fields == {}


# keeping the LLM honest

def test_explanation_with_an_invented_number_is_replaced_by_the_plain_summary():
    chat, _ = chat_with({CAR_TEXT: CAR_FIELDS},
                        "It should sell for 17,250 dollars, a great deal.")
    reply = chat.send(CAR_TEXT)
    assert reply.kind == "estimate"
    assert "17,250" not in reply.text
    assert "$11,600" in reply.text            # the rebuilt-title number, from Python
    assert "2021" in reply.text


def test_llm_failure_while_explaining_still_returns_the_numbers():
    chat, _ = chat_with({CAR_TEXT: CAR_FIELDS}, RuntimeError("API down"))
    reply = chat.send(CAR_TEXT)
    assert reply.kind == "estimate" and reply.estimate.price == 16500
    assert reply.headline.startswith("Estimated asking price: $16,500")


def test_llm_failure_while_extracting_gives_an_error_reply():
    chat, _ = chat_with(RuntimeError("API down"))
    reply = chat.send(CAR_TEXT)
    assert reply.kind == "error" and reply.estimate is None


def test_number_check():
    facts = build_facts(CAR, estimator().estimate(CAR))
    assert adds_no_new_numbers("About 16,500 dollars, range 13,200 to 20,600.", facts)
    assert adds_no_new_numbers("Without the 90,000 miles it would be higher, in 2016.", facts)
    assert adds_no_new_numbers("A rebuilt title takes off 4,950 dollars.", facts)  # sign dropped
    assert not adds_no_new_numbers("Around 16,900 dollars.", facts)
    assert not adds_no_new_numbers("Around 16.5k dollars.", facts)
    assert adds_no_new_numbers("Two years newer, so 3 or 4 steps.", facts)  # small numbers are fine


def test_plain_summary_mentions_assumptions_and_caveats():
    result = estimator().estimate({**CAR, "model": "made up name"})
    facts = build_facts({**CAR, "model": "made up name"}, result,
                        assumptions=["Assuming the title status is clean."])
    text = plain_summary(facts)
    assert "Assuming the title status is clean." in text
    assert "not in the data" in text
