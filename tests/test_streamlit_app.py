from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.app import Chat  # noqa: E402
from tests.test_app import CAR_FIELDS, CAR_TEXT, GOOD_EXPLANATION, ScriptedClient  # noqa: E402
from tests.test_estimate import CFG, estimator  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def app_with(extractions, explanation="Placeholder."):
    """The page with a Chat that uses a fake LLM and fake model, so no key or files needed."""
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["chat"] = Chat(ScriptedClient(extractions, explanation), "fake",
                                    estimator(), CFG)
    return at.run()


def shown(at):
    return " ".join(m.value for m in at.markdown)


def test_page_opens_with_a_greeting_and_no_errors():
    at = app_with({})
    assert not at.exception and not at.error
    assert "Describe a used car" in shown(at) or "Describe a used car" in at.chat_input[0].placeholder


def test_estimate_shows_the_headline_and_escapes_dollar_signs():
    at = app_with({CAR_TEXT: CAR_FIELDS}, GOOD_EXPLANATION)
    at.chat_input[0].set_value(CAR_TEXT).run()
    assert not at.exception
    text = shown(at)
    assert "Estimated asking price: \\$16,500" in text      # escaped for Streamlit
    assert GOOD_EXPLANATION in text
    assert "What moves the price" in [e.label for e in at.expander]


def test_missing_details_show_a_question_and_no_details_box():
    at = app_with({"2016 Camry": dict(year=2016, manufacturer="toyota", model="camry")})
    at.chat_input[0].set_value("2016 Camry").run()
    assert "still need" in shown(at)
    assert not at.expander


def test_start_over_clears_the_conversation():
    at = app_with({CAR_TEXT: CAR_FIELDS})
    at.chat_input[0].set_value(CAR_TEXT).run()
    at.sidebar.button[0].click().run()
    assert at.session_state["chat"].fields == {}
    assert len(at.session_state["messages"]) == 1
