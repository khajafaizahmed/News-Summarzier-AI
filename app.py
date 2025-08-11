import os
import io
import gc
import uuid
import random
import time
import re
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

import streamlit as st
import feedparser
import numpy as np
import requests
from newspaper import Article, Config

# transformers pipeline import (compatible across versions)
try:
    from transformers.pipelines import pipeline
except Exception:
    from transformers import pipeline

from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip
from PIL import Image, ImageDraw, ImageFont

# --- Optional NLTK punkt for newspaper3k ---
try:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
except Exception:
    pass

# Make sure HF token (from Secrets) is visible to libs
HF_TOKEN = st.secrets.get("HUGGINGFACE_HUB_TOKEN", os.getenv("HUGGINGFACE_HUB_TOKEN"))
if HF_TOKEN:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

# --------------------
# Page Setup
# --------------------
st.set_page_config(page_title="Automated News to Video", page_icon="📰", layout="wide")
st.title("📰 Automated News to Video Bot")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.write(
    "Pick regions, language, then generate: the app fetches articles, summarizes with AI, "
    "optionally translates, converts to speech, and produces short captioned videos."
)

# --------------------
# Controls
# --------------------
NEWS_FEEDS = {
    "International": [
        "https://www.reuters.com/world/rss",
        "https://feeds.apnews.com/apf-worldnews",
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://www.theguardian.com/world/rss",
    ],
    "U.S.": [
        "https://www.reuters.com/world/us/rss",
        "https://feeds.apnews.com/apf-usnews",
        "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml",
        "https://feeds.npr.org/1001/rss.xml",
    ],
}

region_choices = st.multiselect(
    "News regions to include",
    options=list(NEWS_FEEDS.keys()),
    default=list(NEWS_FEEDS.keys()),
)

# Low-memory mode
low_mem = st.checkbox("Low‑memory mode (smaller video, fewer items)", value=True)
# Videos per region (default lower to save RAM)
vids = st.slider("Videos per region", 1, 3, 2 if low_mem else 3, help="Reduce if the free tier feels slow.")

# Frame size / fps tuned by mode
if low_mem:
    FRAME_SIZE = (960, 540)
    FRAME_FPS = 20
    # hard-cap how many we actually do in code too
    vids = min(vids, 2)
else:
    FRAME_SIZE = (1280, 720)
    FRAME_FPS = 24

seed_text = st.text_input("Random seed (optional)", help="Enter any value for reproducible picks.")
if seed_text:
    random.seed(seed_text)

LANG_OPTIONS = {
    "English": "en",
    "Spanish": "es",
    "Hindi": "hi",
    "French": "fr",
    "Italian": "it",
}
voice_label = st.selectbox("Voice / Summary language", list(LANG_OPTIONS.keys()), index=0)
VOICE_LANG = LANG_OPTIONS[voice_label]
DO_TRANSLATE = st.checkbox(
    "Translate the summary text to the selected language",
    value=(VOICE_LANG != "en"),
    help="If off, summaries will stay in English even if the voice is set to another language.",
)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# Models (cached)
# --------------------
@st.cache_resource
def load_summarizer():
    try:
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
    except Exception as e:
        st.warning(f"Hugging Face download failed ({type(e).__name__}): {e}. Using a local fallback summary.")
        return None

summarizer = load_summarizer()

# IMPORTANT: translator is NOT cached to keep memory down
def get_translator(lang_code: str):
    if lang_code == "en":
        return None
    model_map = {
        "es": ("translation_en_to_es", "Helsinki-NLP/opus-mt-en-es"),
        "fr": ("translation_en_to_fr", "Helsinki-NLP/opus-mt-en-fr"),
        "it": ("translation_en_to_it", "Helsinki-NLP/opus-mt-en-it"),
        "hi": ("translation_en_to_hi", "Helsinki-NLP/opus-mt-en-hi"),
    }
    tm = model_map.get(lang_code)
    if not tm:
        return None
    task, model_name = tm
    try:
        return pipeline(task, model=model_name, device=-1)
    except Exception as e:
        st.warning(f"Translator load failed ({type(e).__name__}): {e}")
        return None

# --------------------
# Article helpers
# --------------------
DENY_URL_PATTERNS = [r"/news/videos/", r"/news/av/", r"/video/", r"/live/"]
DENY_TEXT_SNIPPETS = ["not responsible for the content of external", "approach to external linking"]

def looks_like_video_or_live(url: str) -> bool:
    return any(re.search(p, url) for p in DENY_URL_PATTERNS)

def is_viable_text(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 600:
        return False
    return not any(s in t.lower() for s in DENY_TEXT_SNIPPETS)

def fetch_html_with_retry(url, retries=2, timeout=20):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; News2VideoBot/1.0; +https://streamlit.app)"}
    for i in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200 and len(r.text) > 500:
                return r.text
        except requests.RequestException:
            pass
        time.sleep(1.5 * (i + 1))
    return None

def fetch_text(url: str) -> str:
    cfg = Config()
    cfg.request_timeout = 20
    cfg.browser_user_agent = "Mozilla/5.0 (compatible; News2VideoBot/1.0)"
    art = Article(url, config=cfg)
    t = ""
    try:
        art.download(); art.parse()
        t = (art.text or "").strip()
    except Exception:
        t = ""
    if not t:
        html = fetch_html_with_retry(url, retries=2, timeout=20)
        if html:
            try:
                art.set_html(html); art.parse()
                t = (art.text or "").strip()
            except Exception:
                t = ""
    return t

def get_random_articles(category_fe
