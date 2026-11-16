import os
from dotenv import load_dotenv
import streamlit as st

# 1. Load .env for local development
load_dotenv()

# 2. Bridge Streamlit Secrets to Environment Variables for the cloud
# This prevents the app from crashing if secrets aren't found locally
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    if "LLAMA_CLOUD_API_KEY" in st.secrets:
        os.environ["LLAMA_CLOUD_API_KEY"] = st.secrets["LLAMA_CLOUD_API_KEY"]
except Exception:
    # This will silently pass if you are running locally and haven't set up .streamlit/secrets.toml
    pass