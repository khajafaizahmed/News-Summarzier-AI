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
    vids = min(vids, 2)  # extra cap in code
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

def get_random_articles(category_feeds, count=3, max_checks=12):
    """Return up to `count` valid articles, skipping video/live/paywalled/external-link placeholders."""
    picks, seen = [], set()
    feeds = list(category_feeds); random.shuffle(feeds)
    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            entries = list(getattr(feed, "entries", [])); random.shuffle(entries)
        except Exception as e:
            st.warning(f"Feed error ({feed_url}): {e}"); continue
        for entry in entries:
            if len(picks) >= count or max_checks <= 0:
                break
            link = getattr(entry, "link", ""); title = getattr(entry, "title", "Untitled")
            source = getattr(feed.feed, "title", "Unknown Source")
            if not link or link in seen or looks_like_video_or_live(link):
                continue
            text = fetch_text(link); max_checks -= 1
            if is_viable_text(text):
                picks.append((link, title, source)); seen.add(link)
        if len(picks) >= count:
            break
    return picks

# --------------------
# Rendering helpers
# --------------------
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
    """
    Language-aware font loader so non-Latin scripts render correctly.
    Looks in assets/ first, then common system paths.
    """
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

    # Symbols fallback (won't cover letters, but can render stray glyphs)
    candidates += [
        os.path.join("assets", "NotoSansSymbols-Regular.ttf"),
        "Arial.ttf",  # last-ditch system fallback
    ]

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

# --------------------
# Core processing
# --------------------
def simple_fallback_summary(text: str) -> str:
    s = re.split(r'(?<=[.!?])\s+', text.strip())
    out, total = [], 0
    for sent in s:
        if not sent: continue
        out.append(sent); total += len(sent)
        if len(out) >= 5 or total > 700: break
    return " ".join(out) if out else text[:600]

def translate_text_safe(text: str, lang: str, translator, do_translate: bool):
    if lang == "en" or not do_translate:
        return text, "en"
    if translator is None:
        return text, "en"
    try:
        # chunk to avoid huge sequence RAM spikes
        chunks = []
        for para in text.split("\n"):
            chunks += [m.group(0).strip() for m in re.finditer(r".{1,500}(?:\s+|$)", para)]
        if not chunks:
            chunks = [text[:500]]
        results = translator(chunks, batch_size=1, max_length=256)
        if isinstance(results, dict):   # just in case a single dict is returned
            results = [results]
        out = " ".join([r.get("translation_text", "").strip() for r in results if r]).strip()
        return (out or text), lang
    except Exception as e:
        st.warning(f"Translation failed ({type(e).__name__}): {e}. Using English instead.")
        return text, "en"

def process_article(url, label, frame_size, frame_fps, translator, do_translate, voice_lang):
    try:
        text = fetch_text(url)
        if not is_viable_text(text):
            return None

        max_len, min_len = (60, 20) if len(text) < 800 else (120, 40)

        if summarizer is not None:
            summary_en = summarizer(
                text[:4000], max_length=max_len, min_length=min_len, do_sample=False
            )[0]["summary_text"]
        else:
            summary_en = simple_fallback_summary(text)

        summary_final, tts_lang = translate_text_safe(summary_en, voice_lang, translator, do_translate)

        uid = uuid.uuid4().hex[:8]
        audio_path = os.path.join(OUTPUT_DIR, f"{label}_{uid}.mp3")
        video_path = os.path.join(OUTPUT_DIR, f"{label}_{uid}.mp4")

        # Cap TTS length a bit to reduce time/memory
        tts_text = summary_final.strip()
        if len(tts_text) > 1200:
            tts_text = tts_text[:1200]

        try:
            tts = gTTS(tts_text, lang=tts_lang)
            tts.save(audio_path)
        except Exception as e:
            st.warning(f"TTS failed: {e}")
            return None

        img_bytes = summary_image(summary_final, size=frame_size, lang=tts_lang)
        frame = np.array(Image.open(img_bytes).convert("RGB"))

        audio = AudioFileClip(audio_path)
        clip = ImageClip(frame).set_duration(audio.duration).set_audio(audio)
        clip.write_videofile(
            video_path,
            fps=frame_fps,
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None
        )

        # Cleanup ASAP to avoid RAM creep
        audio.close()
        clip.close()
        del frame, audio, clip, img_bytes
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

# --------------------
# UI Controls
# --------------------
c1, c2 = st.columns([1, 1])
with c1:
    generate_clicked = st.button("Generate News Videos")
with c2:
    if st.button("Clear output"):
        clear_output()
        st.success("Output folder cleared.")

# --------------------
# Main
# --------------------
if generate_clicked:
    if not region_choices:
        st.warning("Please select at least one news region.")
    else:
        st.info("Fetching & processing… first run may take a couple of minutes (model download).")

        # Load ONE translator for this run, reuse, then free
        translator = get_translator(VOICE_LANG) if (DO_TRANSLATE and VOICE_LANG != "en") else None

        for region in region_choices:
            st.subheader(region)
            feeds = NEWS_FEEDS.get(region, [])
            articles = get_random_articles(feeds, count=vids)
            if not articles:
                st.warning(f"No articles found right now for {region}.")
                continue

            cols = st.columns(3)
            for idx, (link, title, source) in enumerate(articles, start=1):
                col = cols[(idx - 1) % 3]
                with col:
                    with st.spinner(f"Processing {region} {idx}…"):
                        result = process_article(
                            url=link,
                            label=f"{region.lower()}_{idx}",
                            frame_size=FRAME_SIZE,
                            frame_fps=FRAME_FPS,
                            translator=translator,
                            do_translate=DO_TRANSLATE,
                            voice_lang=VOICE_LANG,
                        )
                        if result:
                            video_path, summary_shown = result
                            st.markdown(f"**{title}**  \n_Source: {source}_  \n[Read article]({link})")
                            st.markdown(f"**Summary:** {summary_shown}")
                            st.video(video_path)
                            try:
                                with open(video_path, "rb") as fh:
                                    st.download_button(
                                        label=f"Download Video {idx}",
                                        data=fh,
                                        file_name=os.path.basename(video_path),
                                        mime="video/mp4",
                                        key=f"dl_{region}_{idx}_{os.path.basename(video_path)}",
                                    )
                            except Exception:
                                pass
                        else:
                            st.info("Skipped (paywalled/empty/unparsable).")

        # Free translator memory at end of run
        translator = None
        gc.collect()
