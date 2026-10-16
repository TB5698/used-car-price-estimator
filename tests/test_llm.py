import json
from types import SimpleNamespace

import pytest

from src import llm
from src.validate import MODEL_COLUMNS


class FakeClient:
    """Stands in for the OpenAI client so tests need no API key or network."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_parse_plain_json():
    assert llm.parse_json_object('{"year": 2016}') == {"year": 2016}


def test_parse_json_inside_code_fence_and_words():
    text = 'Sure! Here you go:\n```json\n{"year": 2016, "model": "camry"}\n```'
    assert llm.parse_json_object(text) == {"year": 2016, "model": "camry"}


def test_parse_bad_replies_give_empty_dict():
    assert llm.parse_json_object("I cannot help with that") == {}
    assert llm.parse_json_object("{not json}") == {}
    assert llm.parse_json_object("") == {}
    assert llm.parse_json_object(None) == {}
    assert llm.parse_json_object("[1, 2, 3]") == {}


def test_extract_car_returns_every_field():
    client = FakeClient('{"year": 2016, "manufacturer": "toyota", "model": "camry", "odometer": 90000}')
    result = llm.extract_car("2016 Camry with 90k miles", client, "some-model")
    assert list(result) == MODEL_COLUMNS
    assert result["year"] == 2016
    assert result["odometer"] == 90000
    assert result["drive"] is None


def test_extract_car_ignores_extra_keys():
    client = FakeClient('{"year": 2016, "color": "red", "price": 5000}')
    result = llm.extract_car("red 2016 car", client, "some-model")
    assert "color" not in result and "price" not in result


def test_extract_car_with_unreadable_reply_is_all_none():
    result = llm.extract_car("hello", FakeClient("sorry, no"), "some-model")
    assert list(result) == MODEL_COLUMNS
    assert all(value is None for value in result.values())


def test_extract_car_sends_message_and_model_with_zero_temperature():
    client = FakeClient("{}")
    llm.extract_car("my 2012 civic", client, "model-x")
    call = client.calls[0]
    assert call["model"] == "model-x"
    assert call["temperature"] == 0
    assert call["messages"][-1] == {"role": "user", "content": "my 2012 civic"}
    assert "null" in call["messages"][0]["content"]


def test_explain_estimate_returns_trimmed_text_and_sends_the_facts():
    client = FakeClient("  The estimate is about 12,300 dollars.  ")
    facts = {"price": 12339, "low": 10800, "high": 13900}
    text = llm.explain_estimate(facts, client, "model-x")
    assert text == "The estimate is about 12,300 dollars."
    assert json.loads(client.calls[0]["messages"][-1]["content"]) == facts


def test_missing_api_key_gives_a_clear_error(monkeypatch):
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NEBIUS_API_KEY"):
        llm.get_client()


def test_missing_model_name_gives_a_clear_error(monkeypatch):
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    monkeypatch.delenv("NEBIUS_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="NEBIUS_MODEL"):
        llm.get_model_name()


def test_client_uses_the_nebius_url(monkeypatch):
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    assert str(llm.get_client().base_url) == llm.BASE_URL
