import os
import io
import gc
import uuid
import random
from datetime import datetime

import streamlit as st
import feedparser
import numpy as np
from newspaper import Article, Config
try:
    from transformers.pipelines import pipeline
except Exception:
    from transformers import pipeline
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip
from PIL import Image, ImageDraw, ImageFont

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
    "Fetches random hard & soft news, summarizes with AI, converts to speech, "
    "and generates short videos with on-screen captions."
)

# Controls
vids = st.slider("Videos per category", 1, 3, 3, help="Reduce if the free tier feels slow.")
seed_text = st.text_input("Random seed (optional)", help="Enter any value for reproducible picks.")
if seed_text:
    random.seed(seed_text)

# --------------------
# Paywall-friendly RSS feeds
# --------------------
NEWS_FEEDS = {
    "Hard News": [
        "https://www.reuters.com/world/rss",
        "https://feeds.apnews.com/apf-topnews",
        "https://feeds.npr.org/1004/rss.xml",
        "https://feeds.bbci.co.uk/news/world/rss.xml",
    ],
    "Soft News": [
        "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
        "https://www.theguardian.com/film/rss",
        "https://www.theguardian.com/music/rss",
    ],
}

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# Cache model (lighter than bart-large)
# --------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

# --------------------
# Helpers
# --------------------
def get_random_articles(category_feeds, count=3):
    """Pick random, de-duplicated articles -> (link, title, source)."""
    picks, seen = [], set()
    feeds = list(category_feeds)
    random.shuffle(feeds)
    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            if getattr(feed, "entries", []):
                entry = random.choice(feed.entries)
                link = getattr(entry, "link", "")
                title = getattr(entry, "title", "Untitled")
                source = getattr(feed.feed, "title", "Unknown Source")
                if link and link not in seen:
                    picks.append((link, title, source))
                    seen.add(link)
        except Exception as e:
            st.warning(f"Feed error ({feed_url}): {e}")
        if len(picks) >= count:
            break
    return picks[:count]

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

def process_article(url, label):
    """
    Scrape, summarize, TTS, and make video for an article.
    Returns (video_path, summary) or None.
    """
    try:
        cfg = Config()
        cfg.request_timeout = 10
        cfg.browser_user_agent = "Mozilla/5.0 (compatible; News2VideoBot/1.0)"
        article = Article(url, config=cfg)
        article.download()
        article.parse()

        text = (article.text or "").strip()
        if not text:
            return None
        # Guard against extremely long inputs (speed & truncation)
        text = text[:4000]

        summary = summarizer(text, max_length=120, min_length=40, do_sample=False)[0]["summary_text"]

        uid = uuid.uuid4().hex[:8]
        audio_path = os.path.join(OUTPUT_DIR, f"{label}_{uid}.mp3")
        video_path = os.path.join(OUTPUT_DIR, f"{label}_{uid}.mp4")

        # TTS with guard
        try:
            tts = gTTS(summary)
            tts.save(audio_path)
        except Exception as e:
            st.warning(f"TTS failed: {e}")
            return None

        # Render summary image -> numpy array for ImageClip
        img_bytes = summary_image(summary)
        frame = np.array(Image.open(img_bytes).convert("RGB"))

        audio = AudioFileClip(audio_path)
        clip = ImageClip(frame).set_duration(audio.duration).set_audio(audio)

        clip.write_videofile(
            video_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None,
        )

        # Cleanup
        audio.close()
        clip.close()
        gc.collect()

        return video_path, summary
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

if generate_clicked:
    st.info("Fetching & processing… first run may take a couple of minutes (model download).")

    for category, feeds in NEWS_FEEDS.items():
        st.subheader(category)
        articles = get_random_articles(feeds, count=vids)
        if not articles:
            st.warning("No articles found for this category right now.")
            continue

        cols = st.columns(3)  # 3 across
        for idx, (link, title, source) in enumerate(articles, start=1):
            col = cols[(idx - 1) % 3]
            with col:
                with st.spinner(f"Processing {category} {idx}…"):
                    result = process_article(link, f"{category.lower()}_{idx}")
                    if result:
                        video_path, summary = result
                        st.markdown(f"**{title}**  \n_Source: {source}_  \n[Read article]({link})")
                        st.markdown(f"**Summary:** {summary}")
                        st.video(video_path)
                        try:
                            with open(video_path, "rb") as fh:
                                st.download_button(
                                    label=f"Download Video {idx}",
                                    data=fh,
                                    file_name=os.path.basename(video_path),
                                    mime="video/mp4",
                                    key=f"dl_{category}_{idx}_{os.path.basename(video_path)}",
                                )
                        except Exception:
                            pass
                    else:
                        st.info("Skipped (paywalled/empty/unparsable).")
