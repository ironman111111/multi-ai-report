import hmac
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Multi-AI Report", page_icon="🤖", layout="wide")

# ---------- AI providers (all use the same OpenAI-style API) ----------
PROVIDERS = {
    "Groq (Llama)": {
        "key_name": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
    },
    "Google Gemini": {
        "key_name": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.8-flash",
    },
    "OpenRouter (free)": {
        "key_name": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openrouter/free",
    },
    "OpenAI": {
        "key_name": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
}
SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and concisely."
DB_FILE = "reports.db"

