import os
import io
import gc
import uuid
import random
import time
import re
from datetime import datetime

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

# Try to import Hugging Face login (for token-based auth)
try:
    from huggingface_hub import login as hf_login
except Exception:
    hf_login = None

# Optional: ensure NLTK punkt for newspaper3k on some hosts
try:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
except Exception:
    pass

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
        "https://feeds.npr.org/1001/rss.xml",  # U.S./Top Stories
    ],
}

region_choices = st.multiselect(
    "News regions to include",
    options=list(NEWS_FEEDS.keys()),
    default=list(NEWS_FEEDS.keys()),
)

vids = st.slider("Videos per region", 1, 3, 3, help="Reduce if the free tier feels slow.")
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
    # Use HF token if provided to avoid 429 rate limits
    tok = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if tok and hf_login:
        try:
            hf_login(token=tok)
        except Exception as e:
            st.warning(f"HF login failed: {e}")

    try:
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception as e:
        st.warning(f"Hugging Face download failed ({type(e).__name__}): {e}")
        return None  # we'll use a simple local fallback

summarizer = load_summarizer()

@st.cache_resource
def get_translator(lang_code: str):
    """
    Lazily load a small translation model when needed.
    Returns None for English.
    """
    if lang_code == "en":
        return None
    model_map = {
        "es": "Helsinki-NLP/opus-mt-en-es",
        "fr": "Helsinki-NLP/opus-mt-en-fr",
        "it": "Helsinki-NLP/opus-mt-en-it",
        "hi": "Helsinki-NLP/opus-mt-en-hi",
    }
    model_name = model_map.get(lang_code)
    if not model_name:
        return None
    return pipeline("translation", model=model_name)

# Preload translator if needed (so the first run doesn’t surprise users)
if DO_TRANSLATE and VOICE_LANG != "en":
    with st.spinner("Loading translation model (first time only)…"):
        _ = get_translator(VOICE_LANG)

# --------------------
# Article helpers
# --------------------
DENY_URL_PATTERNS = [
    r"/news/videos/",   # BBC video pages
    r"/news/av/",       # BBC AV pages
    r"/video/",         # generic video pages
    r"/live/",          # live blogs
]
DENY_TEXT_SNIPPETS = [
    "not responsible for the content of external",
    "approach to external linking",
]

def looks_like_video_or_live(url: str) -> bool:
    return any(re.search(p, url) for p in DENY_URL_PATTERNS)

def is_viable_text(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 600:
        return False
    low = t.lower()
    return not any(snip in low for snip in DENY_TEXT_SNIPPETS)

def fetch_html_with_retry(url, retries=2, timeout=20):
    """Fallback fetch for sites that timeout with newspaper.download()."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; News2VideoBot/1.0; +https://streamlit.app)"
    }
    for i in range(retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200 and len(resp.text) > 500:
                return resp.text
        except requests.RequestException:
            pass
        time.sleep(1.5 * (i + 1))  # simple backoff
    return None

def fetch_text(url: str) -> str:
    """Try newspaper3k first; if empty/timeout, retry with requests + set_html."""
    cfg = Config()
    cfg.request_timeout = 20
    cfg.browser_user_agent = "Mozilla/5.0 (compatible; News2VideoBot/1.0)"
    art = Article(url, config=cfg)

    # First try: standard download/parse
    t = ""
    try:
        art.download()
        art.parse()
        t = (art.text or "").strip()
    except Exception:
        t = ""

    # Fallback: requests + set_html
    if not t:
        html = fetch_html_with_retry(url, retries=2, timeout=20)
        if html:
            try:
                art.set_html(html)
                art.parse()
                t = (art.text or "").strip()
            except Exception:
                t = ""
    return t

def get_random_articles(category_feeds, count=3, max_checks=12):
    """Return up to `count` valid articles (skip video/live/paywalled/external-link placeholders)."""
    picks, seen = [], set()
    feeds = list(category_feeds)
    random.shuffle(feeds)

    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            entries = list(getattr(feed, "entries", []))
            random.shuffle(entries)
        except Exception as e:
            st.warning(f"Feed error ({feed_url}): {e}")
            continue

        for entry in entries:
            if len(picks) >= count or max_checks <= 0:
                break

            link = getattr(entry, "link", "")
            title = getattr(entry, "title", "Untitled")
            source = getattr(feed.feed, "title", "Unknown Source")

            if not link or link in seen or looks_like_video_or_live(link):
                continue

            # Validate by fetching text quickly
            text = fetch_text(link)
            max_checks -= 1
            if is_viable_text(text):
                picks.append((link, title, source))
                seen.add(link)

        if len(picks) >= count:
            break

    return picks

# --------------------
# Rendering helpers
# --------------------
def wrap_text_to_fit(draw, text, font, max_width):
    """Wrap text into lines that fit within max_width."""
    words = text.split()
    lines, line = [], []
    for w in words:
        test = " ".join(line + [w])
        l, t, r, b = draw.textbbox((0, 0), test, font=font)
        w_width = r - l
        if w_width <= max_width:
            line.append(w)
        else:
            if line:
                lines.append(" ".join(line))
            line = [w]
    if line:
        lines.append(" ".join(line))
    return lines

def _load_font(size=36):
    """Try bundled/system fonts; fall back to default."""
    candidates = [
        os.path.join("assets", "DejaVuSans.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
        "Arial.ttf",
    ]
    for fp in candidates:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()

def summary_image(summary, size=(1280, 720), margin=80):
    """Render the summary onto a PIL image and return as BytesIO."""
    img = Image.new("RGB", size, (12, 12, 12))
    draw = ImageDraw.Draw(img)

    font = _load_font(36)
    max_text_width = size[0] - 2 * margin
    lines = wrap_text_to_fit(draw, summary, font, max_text_width)

    # Correct line height
    l, t, r, b = draw.textbbox((0, 0), "Ag", font=font)
    line_height = (b - t) + 10
    text_block_height = line_height * len(lines)
    y = (size[1] - text_block_height) // 2

    for line in lines:
        l2, t2, r2, b2 = draw.textbbox((0, 0), line, font=font)
        line_w = r2 - l2
        x = (size[0] - line_w) // 2
        draw.text((x, y), line, font=font, fill=(240, 240, 240))
        y += line_height

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# --------------------
# Core processing
# --------------------
def simple_fallback_summary(text: str) -> str:
    """Very small local fallback if the HF model couldn't load (no extra deps)."""
    # take first ~3-5 sentences up to ~700 chars
    s = re.split(r'(?<=[.!?])\s+', text.strip())
    out, total = [], 0
    for sent in s:
        if not sent:
            continue
        out.append(sent)
        total += len(sent)
        if len(out) >= 5 or total > 700:
            break
    return " ".join(out) if out else text[:600]

def maybe_translate(text: str, target_lang: str, do_translate: bool) -> (str, str):
    """
    Translate `text` to `target_lang` if requested and supported.
    Returns (final_text, final_lang_for_tts).
    If translation fails, falls back to English text + 'en' voice.
    """
    if not do_translate or target_lang == "en":
        return text, "en"
    try:
        translator = get_translator(target_lang)
        if translator is None:
            return text, "en"
        out = translator(text[:2000])[0]["translation_text"]
        if (out or "").strip():
            return out, target_lang
    except Exception as e:
        st.warning(f"Translation failed ({type(e).__name__}): {e}. Using English instead.")
        return text, "en"
    return text, "en"

def process_article(url, label):
    """
    Scrape, summarize, (optional translate), TTS, and make video for an article.
    Returns (video_path, summary_shown) or None.
    """
    try:
        text = fetch_text(url)
        if not is_viable_text(text):
            return None

        # Adjust summary length for short pieces
        char_len = len(text)
        if char_len < 800:
            max_len, min_len = 60, 20
        else:
            max_len, min_len = 120, 40

        if summarizer is not None:
            summary_en = summarizer(
                text[:4000], max_length=max_len, min_length=min_len, do_sample=False
            )[0]["summary_text"]
        else:
            summary_en = simple_fallback_summary(text)

        # Translate if needed so both on-screen text and TTS match the selection
        summary_final, tts_lang = maybe_translate(summary_en, VOICE_LANG, DO_TRANSLATE)

        uid = uuid.uuid4().hex[:8]
        audio_path = os.path.join(OUTPUT_DIR, f"{label}_{uid}.mp3")
        video_path = os.path.join(OUTPUT_DIR, f"{label}_{uid}.mp4")

        # TTS with guard
        try:
            tts = gTTS(summary_final, lang=tts_lang)
            tts.save(audio_path)
        except Exception as e:
            st.warning(f"TTS failed: {e}")
            return None

        # Render summary image -> numpy array for ImageClip
        img_bytes = summary_image(summary_final)
        frame = np.array(Image.open(img_bytes).convert("RGB"))

        audio = AudioFileClip(audio_path)
        clip = ImageClip(frame).set_duration(audio.duration).set_audio(audio)
        clip.write_videofile(
            video_path, fps=24, codec="libx264", audio_codec="aac", verbose=False, logger=None
        )

        audio.close()
        clip.close()
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

        for region in region_choices:
            st.subheader(region)
            feeds = NEWS_FEEDS.get(region, [])
            articles = get_random_articles(feeds, count=vids)
            if not articles:
                st.warning(f"No articles found right now for {region}.")
                continue

            cols = st.columns(3)  # 3 across
            for idx, (link, title, source) in enumerate(articles, start=1):
                col = cols[(idx - 1) % 3]
                with col:
                    with st.spinner(f"Processing {region} {idx}…"):
                        result = process_article(link, f"{region.lower()}_{idx}")
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
