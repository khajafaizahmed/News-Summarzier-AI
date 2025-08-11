import os
import io
import gc
import uuid
import random
import time
import re
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

import streamlit as st
import feedparser
import numpy as np
import requests
from newspaper import Article, Config

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

# --------------------
# Setup folders
# --------------------
OUTPUT_DIR = "output"
ARCHIVE_DIR = "archive"
LOG_FILE = os.path.join(ARCHIVE_DIR, "log.jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# --------------------
# HF Token
# --------------------
HF_TOKEN = st.secrets.get("HUGGINGFACE_HUB_TOKEN", os.getenv("HUGGINGFACE_HUB_TOKEN"))
if HF_TOKEN:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

# --------------------
# Page Setup
# --------------------
st.set_page_config(page_title="Automated News to Video", page_icon="📰", layout="wide")
st.title("📰 Automated News to Video Bot")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --------------------
# Feeds
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

# --------------------
# Controls
# --------------------
region_choices = st.multiselect(
    "Which regions to generate? (one video per region)",
    options=list(NEWS_FEEDS.keys()),
    default=list(NEWS_FEEDS.keys()),
)

seed_text = st.text_input("Random seed (optional)")
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
)

# --------------------
# Models
# --------------------
@st.cache_resource
def load_summarizer():
    try:
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
    except Exception as e:
        st.warning(f"Hugging Face download failed: {e}")
        return None

summarizer = load_summarizer()

@st.cache_resource
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
    try:
        return pipeline(tm[0], model=tm[1], device=-1)
    except Exception as e:
        st.warning(f"Translator load failed: {e}")
        return None

if DO_TRANSLATE and VOICE_LANG != "en":
    with st.spinner("Loading translation model…"):
        _ = get_translator(VOICE_LANG)

# --------------------
# Article fetching
# --------------------
DENY_URL_PATTERNS = [r"/news/videos/", r"/news/av/", r"/video/", r"/live/"]
DENY_TEXT_SNIPPETS = ["not responsible for the content of external", "approach to external linking"]

def looks_like_video_or_live(url: str) -> bool:
    return any(re.search(p, url) for p in DENY_URL_PATTERNS)

def is_viable_text(text: str) -> bool:
    return len((text or "").strip()) >= 600 and not any(s in text.lower() for s in DENY_TEXT_SNIPPETS)

def fetch_html_with_retry(url, retries=2, timeout=20):
    headers = {"User-Agent": "Mozilla/5.0"}
    for i in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200 and len(r.text) > 500:
                return r.text
        except:
            pass
        time.sleep(1.2 * (i + 1))
    return None

def fetch_text(url: str) -> str:
    cfg = Config()
    cfg.request_timeout = 20
    cfg.browser_user_agent = "Mozilla/5.0"
    art = Article(url, config=cfg)
    try:
        art.download(); art.parse()
        return (art.text or "").strip()
    except:
        html = fetch_html_with_retry(url)
        if html:
            try:
                art.set_html(html); art.parse()
                return (art.text or "").strip()
            except:
                return ""
    return ""

def pick_one_article(region_name: str):
    feeds = list(NEWS_FEEDS.get(region_name, []))
    random.shuffle(feeds)
    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            entries = list(feed.entries)
            random.shuffle(entries)
        except:
            continue
        for entry in entries:
            link = getattr(entry, "link", "")
            title = getattr(entry, "title", "Untitled")
            source = getattr(feed.feed, "title", "Unknown Source")
            if link and not looks_like_video_or_live(link):
                text = fetch_text(link)
                if is_viable_text(text):
                    return (link, title, source)
    return None

# --------------------
# Rendering helpers
# --------------------
def wrap_text_to_fit(draw, text, font, max_width):
    words = text.split()
    lines, line = [], []
    for w in words:
        test = " ".join(line + [w])
        if draw.textbbox((0,0), test, font=font)[2] <= max_width:
            line.append(w)
        else:
            if line: lines.append(" ".join(line))
            line = [w]
    if line: lines.append(" ".join(line))
    return lines

def _load_font(size=36, lang="en"):
    if lang == "hi":
        candidates = ["assets/NotoSansDevanagari-Regular.ttf"]
    else:
        candidates = ["assets/NotoSans-Regular.ttf", "assets/DejaVuSans.ttf"]
    for fp in candidates:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()

def summary_image(summary, size=(1280,720), margin=80, lang="en"):
    img = Image.new("RGB", size, (12, 12, 12))
    draw = ImageDraw.Draw(img)
    font = _load_font(36, lang)
    lines = wrap_text_to_fit(draw, summary, font, size[0] - 2*margin)
    y = margin
    for line in lines:
        w = draw.textbbox((0,0), line, font=font)[2]
        x = (size[0] - w)//2
        draw.text((x,y), line, font=font, fill=(240,240,240))
        y += 46
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# --------------------
# Core
# --------------------
def process_article(url, label, region, title):
    text = fetch_text(url)
    if not is_viable_text(text):
        return None
    if summarizer:
        summary_en = summarizer(text[:4000], max_length=120, min_length=40, do_sample=False)[0]["summary_text"]
    else:
        summary_en = text[:600]
    summary_final, tts_lang = translate_text_safe(summary_en, VOICE_LANG)
    uid = uuid.uuid4().hex[:8]
    audio_path = os.path.join(OUTPUT_DIR, f"{label}_{uid}.mp3")
    video_path = os.path.join(OUTPUT_DIR, f"{label}_{uid}.mp4")
    gTTS(summary_final, lang=tts_lang).save(audio_path)
    img_bytes = summary_image(summary_final, lang=tts_lang)
    frame = np.array(Image.open(img_bytes).convert("RGB"))
    audio = AudioFileClip(audio_path)
    clip = ImageClip(frame).set_duration(audio.duration).set_audio(audio)
    clip.write_videofile(video_path, fps=24, codec="libx264", audio_codec="aac", verbose=False, logger=None)
    audio.close(); clip.close(); gc.collect()

    # Save permanent copy
    archive_video = os.path.join(ARCHIVE_DIR, os.path.basename(video_path))
    archive_audio = os.path.join(ARCHIVE_DIR, os.path.basename(audio_path))
    os.replace(video_path, archive_video)
    os.replace(audio_path, archive_audio)
    with open(LOG_FILE, "a", encoding="utf-8") as logf:
        logf.write(json.dumps({
            "region": region,
            "title": title,
            "language": VOICE_LANG,
            "summary": summary_final,
            "video_file": os.path.basename(archive_video),
            "audio_file": os.path.basename(archive_audio),
            "timestamp": datetime.now().isoformat()
        }) + "\n")
    return archive_video, summary_final

def translate_text_safe(text, lang):
    if lang == "en" or not DO_TRANSLATE:
        return text, "en"
    tr = get_translator(lang)
    if not tr:
        return text, "en"
    chunks = [text[i:i+400] for i in range(0, len(text), 400)]
    results = tr(chunks, batch_size=1, max_length=256)
    if isinstance(results, dict):
        results = [results]
    out = " ".join([r.get("translation_text", "") for r in results])
    return out.strip() or text, lang

def clear_output():
    for f in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, f))

# --------------------
# UI
# --------------------
c1, c2 = st.columns([1,1])
with c1:
    generate_clicked = st.button("Generate Videos (one per region)")
with c2:
    if st.button("Clear temp output"):
        clear_output()
        st.success("Temporary output cleared.")

if generate_clicked:
    for region in region_choices:
        st.subheader(region)
        pick = pick_one_article(region)
        if not pick:
            st.warning(f"No article found for {region}.")
            continue
        link, title, source = pick
        with st.spinner("Processing..."):
            result = process_article(link, region.lower(), region, title)
        if result:
            video_path, summary_shown = result
            st.markdown(f"**{title}**\n\n_Source: {source}_\n\n[Read article]({link})")
            st.markdown(f"**Summary:** {summary_shown}")
            st.video(video_path)
        else:
            st.info("Skipped.")
