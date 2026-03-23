import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)


def get_secret(name):
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    if value is None:
        value = os.getenv(name)
    return value


def load_api_key():
    api_key = get_secret("OPENAI_API_KEY")
    if api_key:
        return api_key

    if not ENV_PATH.exists():
        return None

    for line in ENV_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().startswith("OPENAI_API_KEY="):
            _, raw_value = line.split("=", 1)
            cleaned = raw_value.strip().strip("\"'“”")
            if cleaned:
                os.environ["OPENAI_API_KEY"] = cleaned
                return cleaned

    return None


OPENAI_API_KEY = load_api_key()
APP_USERNAME = get_secret("APP_USERNAME")
APP_PASSWORD = get_secret("APP_PASSWORD")
