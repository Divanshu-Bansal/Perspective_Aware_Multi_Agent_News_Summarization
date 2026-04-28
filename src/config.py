import os
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY     = os.getenv("NEWS_API_KEY")
GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY")
GNEWS_API_KEY    = os.getenv("GNEWS_API_KEY")

CLEARML_API_ACCESS_KEY = os.getenv("CLEARML_API_ACCESS_KEY")
CLEARML_API_SECRET_KEY = os.getenv("CLEARML_API_SECRET_KEY")
CLEARML_API_HOST       = os.getenv("CLEARML_API_HOST", "https://api.clear.ml")