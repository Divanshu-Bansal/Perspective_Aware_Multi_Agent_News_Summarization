import os
from dotenv import load_dotenv

# Load environment variables from .env file (local development)
load_dotenv()


def _get(key: str) -> str:
    """
    Retrieves configuration values from environment variables
    or Streamlit secrets (for deployed environments).

    Priority:
    1. Environment variables (.env / system env)
    2. Streamlit secrets (for cloud deployment)
    3. Fallback to empty string if not found

    Args:
        key (str): Configuration key name

    Returns:
        str: Retrieved value or empty string if not available
    """

    # Attempt to read from environment variables first
    value = os.getenv(key)


    if not value:
        try:
            # Attempt to read from Streamlit secrets (if running in Streamlit)
            import streamlit as st
            value = st.secrets.get(key, "")
        except Exception:
            # Ignore errors (e.g., Streamlit not installed in local environment)
            pass

    return value or ""


# ─────────────────────────────────────────────────────────────
# API Keys & External Service Configuration
# ─────────────────────────────────────────────────────────────

# News APIs
NEWS_API_KEY           = _get("NEWS_API_KEY")
GUARDIAN_API_KEY       = _get("GUARDIAN_API_KEY")
GNEWS_API_KEY          = _get("GNEWS_API_KEY")

# ClearML configuration (experiment tracking)
CLEARML_API_ACCESS_KEY = _get("CLEARML_API_ACCESS_KEY")
CLEARML_API_SECRET_KEY = _get("CLEARML_API_SECRET_KEY")

# Default ClearML API host (fallback if not provided)
CLEARML_API_HOST       = _get("CLEARML_API_HOST") or "https://api.clear.ml"