import os
from dotenv import load_dotenv

load_dotenv()

def _get(key: str) -> str:
    """Get config value from environment or Streamlit secrets."""
    value = os.getenv(key)
    if not value:
        try:
            import streamlit as st
            value = st.secrets.get(key, "")
        except Exception:
            pass
    return value or ""

NEWS_API_KEY           = _get("NEWS_API_KEY")
GUARDIAN_API_KEY       = _get("GUARDIAN_API_KEY")
GNEWS_API_KEY          = _get("GNEWS_API_KEY")
CLEARML_API_ACCESS_KEY = _get("CLEARML_API_ACCESS_KEY")
CLEARML_API_SECRET_KEY = _get("CLEARML_API_SECRET_KEY")
CLEARML_API_HOST       = _get("CLEARML_API_HOST") or "https://api.clear.ml"