"""Chat page for the used car price estimator.

Run from the repo root with the venv active:
    streamlit run streamlit_app.py

All the logic lives in src/app.py. This file only draws the conversation.
"""
import streamlit as st

from src.app import Chat
from src.estimate import PriceEstimator
from src.llm import get_client, get_model_name
from src.preprocess import load_config

GREETING = (
    "Hi! Describe a used car and I'll estimate its asking price. Include the year, make, "
    "model, mileage, body type, drive, fuel and transmission, for example: "
    "*2016 Toyota Camry, 90k miles, sedan, front wheel drive, gas, automatic*."
)

st.set_page_config(page_title="Used Car Price Estimator", page_icon="🚗")


def safe(text):
    """Streamlit reads dollar signs as math, so escape them."""
    return text.replace("$", "\\$")


@st.cache_resource(show_spinner="Loading the model...")
def load_shared():
    """The LLM client and the price model are loaded once and shared by all sessions."""
    return get_client(), get_model_name(), PriceEstimator.load(), load_config()


def new_chat():
    return Chat(*load_shared())


def money(amount):
    return safe(f"${amount:,}")


def show_details(facts):
    """The numbers behind an estimate, straight from Python (not the LLM)."""
    moves = facts["what_moves_the_price"]
    if not moves:
        return
    with st.expander("What moves the price"):
        for move in moves:
            change = move["price_change"]
            sign = "+" if change > 0 else "-"
            st.markdown(f"- **{move['if_instead']}**: {money(move['new_estimate'])} "
                        f"({sign}{money(abs(change))})")
        basis = facts["price_basis"]
        st.caption(basis[0].upper() + basis[1:] + ".")


def show_message(message):
    with st.chat_message(message["role"]):
        if message.get("headline"):
            st.markdown(f"**{safe(message['headline'])}**")
        st.markdown(safe(message["text"]))
        if message.get("facts"):
            show_details(message["facts"])


st.title("🚗 Used Car Price Estimator")
st.caption("Typical asking prices from 2021 Craigslist listings, for everyday cars "
           "from 1990 and newer.")

try:
    if "chat" not in st.session_state:
        st.session_state.chat = new_chat()
except Exception as error:  # missing key, model name or model files
    st.error(f"The app could not start: {error}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "text": GREETING}]

with st.sidebar:
    st.header("About")
    st.write(
        "A language model reads your description, Python checks it, and a machine "
        "learning model (LightGBM trained on about 364,000 listings) prices the car. "
        "The language model only explains the numbers; it never makes them up."
    )
    if st.button("Start over"):
        st.session_state.chat.reset()
        st.session_state.messages = [{"role": "assistant", "text": GREETING}]
        st.rerun()

for message in st.session_state.messages:
    show_message(message)

prompt = st.chat_input("Describe a used car")
if prompt:
    st.session_state.messages.append({"role": "user", "text": prompt})
    show_message(st.session_state.messages[-1])
    with st.spinner("Thinking..."):
        reply = st.session_state.chat.send(prompt)
    st.session_state.messages.append({
        "role": "assistant", "text": reply.text,
        "headline": reply.headline, "facts": reply.facts,
    })
    show_message(st.session_state.messages[-1])
