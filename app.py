import os
import io
import gc
import uuid
import random
import time
import re
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=SyntaxWarning)

import streamlit as st
import feedparser
import numpy as np
import requests
from newspaper import Article, Config

# transformers import (robust to environments where `transformers.pipelines` isn't exposed)
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

# -----------------------------------------------------------------------------
# Folders (no archive; only a temp output folder kept out of git)
# -----------------------------------------------------------------------------
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Hugging Face token (optional; reduces 429s)
# -----------------------------------------------------------------------------
HF_TOKEN = st.secrets.get("HUGGINGFACE_HUB_TOKEN", os.getenv("HUGGINGFACE_HUB_TOKEN"))
if HF_TOKEN:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Automated News to Video", page_icon="📰", layout="wide")
st.title("📰 Automated News to Video Bot")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.write(
    "Pick regions + language, then generate up to two short videos (one per region). "
    "The app fetches an article, summarizes with AI, optionally translates, converts to speech, "
    "and renders on-screen captions."
)

# -----------------------------------------------------------------------------
# Feeds (paywall‑friendly)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Controls
# -----------------------------------------------------------------------------
region_choices = st.multiselect(
    "Which regions to generate? (one video per region)",
    options=list(NEWS_FEEDS.keys()),
    default=list(NEWS_FEEDS.keys()),  # both
)

seed_text = st.text_input("Random seed (optional)", help="Enter any value for reproducible picks.")
if seed_text:
    random.seed(seed_text)

LANG_OPTIONS = {"English": "en", "Spanish": "es", "Hindi": "hi", "French": "fr", "Italian": "it"}
voice_label = st.selectbox("Voice / Summary language", list(LANG_OPTIONS.keys()), index=0)
VOICE_LANG = LANG_OPTIONS[voice_label]
DO_TRANSLATE = st.checkbox(
    "Translate the summary text to the selected language",
    value=(VOICE_LANG != "en"),
    help="If off, summaries will stay in English even if the voice is set to another language.",
)

# -----------------------------------------------------------------------------
# Models (cached, CPU-only to stay in Community limits)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_summarizer():
    try:
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
    except Exception as e:
        st.warning(f"Hugging Face summarizer load failed ({type(e).__name__}): {e}. Using a simple fallback.")
        return None

summarizer = load_summarizer()

@st.cache_resource
def get_translator(lang_code: str):
    if lang_code == "en":
        return None
    mm = {
        "es": ("translation_en_to_es", "Helsinki-NLP/opus-mt-en-es"),
        "fr": ("translation_en_to_fr", "Helsinki-NLP/opus-mt-en-fr"),
        "it": ("translation_en_to_it", "Helsinki-NLP/opus-mt-en-it"),
        "hi": ("translation_en_to_hi", "Helsinki-NLP/opus-mt-en-hi"),
    }
    pair = mm.get(lang_code)
    if not pair:
        return None
    task, model = pair
    try:
        return pipeline(task, model=model, device=-1)
    except Exception as e:
        st.warning(f"Translator load failed ({type(e).__name__}): {e}")
        return None

# Lazy pre-load to avoid surprise delay on first run
if DO_TRANSLATE and VOICE_LANG != "en":
    with st.spinner("Loading translation model (first time only)…"):
        _ = get_translator(VOICE_LANG)

# -----------------------------------------------------------------------------
# Article helpers
# -----------------------------------------------------------------------------
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
    headers = {"User-Agent": "Mozilla/5.0 (compatible; News2VideoBot/1.0)"}
    for i in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200 and len(r.text) > 500:
                return r.text
        except requests.RequestException:
            pass
        time.sleep(1.2 * (i + 1))
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

def pick_one_article(region_name: str):
    feeds = list(NEWS_FEEDS.get(region_name, []))
    random.shuffle(feeds)
    checks_left = 10
    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            entries = list(getattr(feed, "entries", []))
            random.shuffle(entries)
        except Exception:
            continue
        for entry in entries:
            if checks_left <= 0:
                break
            link = getattr(entry, "link", "")
            title = getattr(entry, "title", "Untitled")
            source = getattr(feed.feed, "title", "Unknown Source")
            if not link or looks_like_video_or_live(link):
                continue
            text = fetch_text(link)
            checks_left -= 1
            if is_viable_text(text):
                return (link, title, source)
    return None

# -----------------------------------------------------------------------------
# Rendering helpers (fonts)
# -----------------------------------------------------------------------------
def wrap_text_to_fit(draw, text, font, max_width):
    words = text.split()
    lines, line = [], []
    for w in words:
        test = " ".join(line + [w])
        l, t, r, b = draw.textbbox((0, 0), test, font=font)
        if (r - l) <= max_width:
            line.append(w)
        else:
            if line: lines.append(" ".join(line))
            line = [w]
    if line: lines.append(" ".join(line))
    return lines

def _load_font(size=36, lang="en"):
    # Expected in assets/: NotoSans-Regular.ttf, NotoSansDevanagari-Regular.ttf, (optional) DejaVuSans.ttf
    if lang == "hi":
        candidates = [
            os.path.join("assets", "NotoSansDevanagari-Regular.ttf"),
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        ]
    else:
        candidates = [
            os.path.join("assets", "NotoSans-Regular.ttf"),
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            os.path.join("assets", "DejaVuSans.ttf"),
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    candidates += ["Arial.ttf"]  # last-ditch
    for fp in candidates:
        try:
            if fp and os.path.exists(fp):
                return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()

def summary_image(summary, size=(1280, 720), margin=80, lang="en"):
    img = Image.new("RGB", size, (12, 12, 12))
    draw = ImageDraw.Draw(img)
    font = _load_font(36, lang=lang)
    max_text_width = size[0] - 2 * margin
    lines = wrap_text_to_fit(draw, summary, font, max_text_width)
    l, t, r, b = draw.textbbox((0, 0), "Ag", font=font)
    line_height = (b - t) + 10
    text_block_height = line_height * max(1, len(lines))
    y = max(margin, (size[1] - text_block_height) // 2)
    for line in lines:
        l2, t2, r2, b2 = draw.textbbox((0, 0), line, font=font)
        x = max(margin, (size[0] - (r2 - l2)) // 2)
        draw.text((x, y), line, font=font, fill=(240, 240, 240))
        y += line_height
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    return buf

# -----------------------------------------------------------------------------
# Core: translation + processing
# -----------------------------------------------------------------------------
def simple_fallback_summary(text: str) -> str:
    s = re.split(r'(?<=[.!?])\s+', text.strip())
    out, total = [], 0
    for sent in s:
        if not sent: continue
        out.append(sent); total += len(sent)
        if len(out) >= 5 or total > 700: break
    return " ".join(out) if out else text[:600]

def translate_text_safe(text: str, lang: str):
    if lang == "en" or not DO_TRANSLATE:
        return text, "en"
    tr = get_translator(lang)
    if tr is None:
        return text, "en"
    try:
        chunks = []
        for para in text.split("\n"):
            chunks += [m.group(0).strip() for m in re.finditer(r".{1,400}(?:\s+|$)", para)]
        if not chunks:
            chunks = [text[:400]]
        results = tr(chunks, batch_size=1, max_length=256)
        if isinstance(results, dict):
            results = [results]
        out = " ".join([r.get("translation_text", "").strip() for r in results if r]).strip()
        return (out or text), lang
    except Exception as e:
        st.warning(f"Translation failed ({type(e).__name__}): {e}. Using English instead.")
        return text, "en"

def estimate_eta_seconds(will_translate: bool, summary_words_guess: int = 80) -> int:
    """
    Tiny heuristic ETA (in seconds). This is intentionally conservative and cheap.
    - Fetch/parse: ~5s
    - Summarize: ~8s
    - Translate (optional): +6s
    - TTS+encode: speech length (~summary_words/160 wpm) + 4s overhead
    """
    fetch_parse = 5
    summarize = 8 if summarizer is not None else 2
    translate = 6 if will_translate else 0
    speech_secs = int((summary_words_guess / 160.0) * 60)  # ~160wpm
    overhead = 4
    return fetch_parse + summarize + translate + speech_secs + overhead

def process_article(url, label):
    try:
        text = fetch_text(url)
        if not is_viable_text(text):
            return None

        # Summarize
        max_len, min_len = (60, 20) if len(text) < 800 else (120, 40)
        if summarizer is not None:
            summary_en = summarizer(
                text[:4000], max_length=max_len, min_length=min_len, do_sample=False
            )[0]["summary_text"]
        else:
            summary_en = simple_fallback_summary(text)

        # Translate if needed
        summary_final, tts_lang = translate_text_safe(summary_en, VOICE_LANG)

        # TTS + video
        uid = uuid.uuid4().hex[:8]
        audio_path = os.path.join(OUTPUT_DIR, f"{label}_{uid}.mp3")
        video_path = os.path.join(OUTPUT_DIR, f"{label}_{uid}.mp4")

        tts = gTTS(summary_final, lang=tts_lang)
        tts.save(audio_path)

        img_bytes = summary_image(summary_final, lang=tts_lang)
        frame = np.array(Image.open(img_bytes).convert("RGB"))

        audio = AudioFileClip(audio_path)
        clip = ImageClip(frame).set_duration(audio.duration).set_audio(audio)
        clip.write_videofile(
            video_path, fps=24, codec="libx264", audio_codec="aac", verbose=False, logger=None
        )

        audio.close(); clip.close()
        del frame
        gc.collect()

        return video_path, summary_final
    except Exception as e:
        st.warning(f"Error processing article: {e}")
        return None

def clear_output():
    try:
        for f in os.listdir(OUTPUT_DIR):
            os.remove(os.path.join(OUTPUT_DIR, f))
    except Exception:
        pass

# -----------------------------------------------------------------------------
# UI buttons
# -----------------------------------------------------------------------------
c1, c2 = st.columns([1, 1])
with c1:
    generate_clicked = st.button("Generate Videos (one per region)")
with c2:
    if st.button("Clear output"):
        clear_output()
        st.success("Output folder cleared.")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if generate_clicked:
    overall_start = time.time()

    if not region_choices:
        st.warning("Please select at least one news region.")
    else:
        st.info("Fetching & processing… first run may take a couple of minutes (model download).")

        # Do regions sequentially (keeps RAM low)
        for region in region_choices:
            region_start = time.time()
            st.subheader(region)

            # Quick ETA before we start (cheap guess)
            eta_guess = estimate_eta_seconds(will_translate=(DO_TRANSLATE and VOICE_LANG != "en"))
            st.caption(f"Estimated time for {region}: ~{eta_guess}s")

            pick = pick_one_article(region)
            if not pick:
                st.warning(f"No viable article found for {region}.")
                continue

            link, title, source = pick
            with st.spinner("Summarizing, translating (if selected), and rendering video…"):
                result = process_article(link, f"{region.lower()}")

            if result:
                video_path, summary_shown = result
                st.markdown(f"**{title}**  \n_Source: {source}_  \n[Read article]({link})")
                st.markdown(f"**Summary:** {summary_shown}")
                st.video(video_path)
                try:
                    with open(video_path, "rb") as fh:
                        st.download_button(
                            label=f"Download {region} Video",
                            data=fh,
                            file_name=os.path.basename(video_path),
                            mime="video/mp4",
                            key=f"dl_{region}_{os.path.basename(video_path)}",
                        )
                except Exception:
                    pass
            else:
                st.info("Skipped (paywalled/empty/unparsable).")

            st.caption(f"⏱️ {region} video completed in {time.time() - region_start:.1f} seconds.")
            gc.collect()

        st.caption(f"✅ All done in {time.time() - overall_start:.1f} seconds.")
