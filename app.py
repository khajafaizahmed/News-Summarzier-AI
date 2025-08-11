import os
import io
import gc
import uuid
import json
import shutil
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

try:
    from transformers.pipelines import pipeline
except Exception:
    from transformers import pipeline

from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip
from PIL import Image, ImageDraw, ImageFont

# --------------------
# Constants / Paths
# --------------------
OUTPUT_DIR = "output"            # temp spot for freshly generated files
ARCHIVE_DIR = "archive"          # durable storage (kept between runs)
ARCHIVE_LOG = os.path.join(ARCHIVE_DIR, "log.jsonl")  # one JSON per line

# --------------------
# Ensure folders exist (safe on Streamlit Cloud)
# --------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)
if not os.path.exists(ARCHIVE_LOG):
    # create an empty file so Streamlit doesn't complain
    with open(ARCHIVE_LOG, "a", encoding="utf-8") as _:
        pass

# --------------------
# Optional NLTK punkt for newspaper3k
# --------------------
try:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
except Exception:
    pass

# Make sure HF token (from Secrets) is visible to libs (avoids 429 rate limit)
HF_TOKEN = st.secrets.get("HUGGINGFACE_HUB_TOKEN", os.getenv("HUGGINGFACE_HUB_TOKEN"))
if HF_TOKEN:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

# --------------------
# Page Setup
# --------------------
st.set_page_config(page_title="Automated News → Video", page_icon="📰", layout="wide")
st.title("📰 Automated News → Video")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Tabs
tab_make, tab_archive = st.tabs(["🎬 Create", "📂 Archive"])

# --------------------
# Feeds (paywall-friendly)
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
    """Pick a single viable article from a region or return None."""
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
    Expected in assets/:
      - NotoSans-Regular.ttf            (Latin for en/es/fr/it)
      - NotoSansDevanagari-Regular.ttf  (Hindi)
      - NotoSansSymbols-Regular.ttf     (fallback symbols)
      - (Optional) DejaVuSans.ttf       (extra Latin coverage)
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

    candidates += [
        os.path.join("assets", "NotoSansSymbols-Regular.ttf"),
        "Arial.ttf",
    ]

    for fp in candidates:
        try:
            if fp and (fp.endswith(".ttf") or fp.endswith(".otf")) and os.path.exists(fp):
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

def translate_text_safe(text: str, lang: str, do_translate: bool):
    if lang == "en" or not do_translate:
        return text, "en"
    tr = get_translator(lang)
    if tr is None:
        return text, "en"
    try:
        # small chunks for low RAM
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

def process_article(url, label, voice_lang, do_translate, meta_region, meta_source, meta_title):
    try:
        text = fetch_text(url)
        if not is_viable_text(text):
            return None
        max_len, min_len = (60, 20) if len(text) < 800 else (120, 40)

        if summarizer is not None:
            summary_en = summarizer(text[:4000], max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
        else:
            summary_en = simple_fallback_summary(text)

        summary_final, tts_lang = translate_text_safe(summary_en, voice_lang, do_translate)

        uid = uuid.uuid4().hex[:8]
        audio_path = os.path.join(OUTPUT_DIR, f"{label}_{uid}.mp3")
        video_path = os.path.join(OUTPUT_DIR, f"{label}_{uid}.mp4")

        # TTS
        try:
            tts = gTTS(summary_final, lang=tts_lang)
            tts.save(audio_path)
        except Exception as e:
            st.warning(f"TTS failed: {e}")
            return None

        # Image → Video
        img_bytes = summary_image(summary_final, lang=tts_lang)
        frame = np.array(Image.open(img_bytes).convert("RGB"))
        audio = AudioFileClip(audio_path)
        clip = ImageClip(frame).set_duration(audio.duration).set_audio(audio)
        clip.write_videofile(video_path, fps=24, codec="libx264", audio_codec="aac", verbose=False, logger=None)

        # cleanup temp objects
        audio.close(); clip.close(); del frame
        gc.collect()

        # Archive (copy, then log)
        archived_path = archive_save(
            video_path,
            {
                "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "region": meta_region,
                "source": meta_source,
                "title": meta_title,
                "lang": tts_lang,
                "summary": summary_final,
                "orig_url": url,
            },
        )

        return archived_path, summary_final
    except Exception as e:
        st.warning(f"Error processing article: {e}")
        return None

# --------------------
# Archive helpers
# --------------------
def safe_name(s: str, maxlen=60):
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "_", s).strip("_")
    return s[:maxlen] if s else "item"

def archive_save(src_video_path: str, meta: dict) -> str:
    """Copy video into archive/ and append metadata to log.jsonl. Return archived path."""
    uid = uuid.uuid4().hex[:8]
    base = f"{meta.get('ts', datetime.utcnow().isoformat(timespec='seconds'))}_{safe_name(meta.get('region','unknown'))}_{uid}.mp4"
    dst = os.path.join(ARCHIVE_DIR, base)
    try:
        shutil.copyfile(src_video_path, dst)
    except Exception:
        # if copy fails, at least keep original
        dst = src_video_path

    meta_to_write = dict(meta)
    meta_to_write["file"] = os.path.relpath(dst)  # relative path for Streamlit
    with open(ARCHIVE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta_to_write, ensure_ascii=False) + "\n")
    return dst

def archive_read_all():
    items = []
    try:
        with open(ARCHIVE_LOG, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # skip entries whose file is missing
                    if obj.get("file") and os.path.exists(obj["file"]):
                        items.append(obj)
                except Exception:
                    continue
    except Exception:
        pass
    # newest first by timestamp string
    items.sort(key=lambda x: x.get("ts", ""), reverse=True)
    return items

def clear_output_temp():
    try:
        for f in os.listdir(OUTPUT_DIR):
            os.remove(os.path.join(OUTPUT_DIR, f))
    except Exception:
        pass

# =========================
# ========  UI  ===========
# =========================
with tab_make:
    st.write(
        "Pick regions + language, then generate up to two short videos (one per region). "
        "We’ll auto‑save videos into the Archive tab."
    )

    # Controls
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

    c1, c2 = st.columns([1, 1])
    with c1:
        generate_clicked = st.button("Generate Videos (one per region)", type="primary")
    with c2:
        if st.button("Clear temp output"):
            clear_output_temp()
            st.success("Temp output folder cleared.")

    if generate_clicked:
        if not region_choices:
            st.warning("Please select at least one news region.")
        else:
            st.info("Fetching & processing… first run may take a couple of minutes (model download).")

            # Generate sequentially per region to keep memory steady
            for region in region_choices:
                region_start = time.time()
                st.subheader(region)

                pick = pick_one_article(region)
                if not pick:
                    st.warning(f"No viable article found for {region}.")
                    continue

                link, title, source = pick
                with st.spinner("Summarizing, translating (if selected), and rendering video…"):
                    result_path = process_article(
                        url=link,
                        label=region.lower(),
                        voice_lang=VOICE_LANG,
                        do_translate=DO_TRANSLATE,
                        meta_region=region,
                        meta_source=source,
                        meta_title=title,
                    )

                if result_path:
                    st.markdown(f"**{title}**  \n_Source: {source}_  \n[Read article]({link})")
                    # Read last line from archive for the summary (avoid re-opening full video)
                    # but we already returned summary inside archive_save -> not needed to reload here
                    st.video(result_path)
                    try:
                        with open(result_path, "rb") as fh:
                            st.download_button(
                                label=f"Download {region} Video",
                                data=fh,
                                file_name=os.path.basename(result_path),
                                mime="video/mp4",
                                key=f"dl_{region}_{os.path.basename(result_path)}",
                            )
                    except Exception:
                        pass
                else:
                    st.info("Skipped (paywalled/empty/unparsable).")

                st.caption(f"⏱️ {region} video completed in {time.time() - region_start:.1f} seconds.")
                gc.collect()

with tab_archive:
    st.write("Your saved videos appear here. Filter and replay without regenerating.")
    all_items = archive_read_all()

    if not all_items:
        st.info("No items in the archive yet. Generate something in the Create tab!")
    else:
        # Filters
        languages = ["any"] + sorted({i.get("lang", "en") for i in all_items})
        regions = ["any"] + sorted({i.get("region", "Unknown") for i in all_items})
        f1, f2, f3 = st.columns([1, 1, 1])
        with f1:
            lang_filter = st.selectbox("Language", languages, index=0)
        with f2:
            region_filter = st.selectbox("Region", regions, index=0)
        with f3:
            show_n = st.number_input("Show latest N", min_value=1, max_value=20, value=5, step=1)

        shown = 0
        for item in all_items:
            if lang_filter != "any" and item.get("lang") != lang_filter:
                continue
            if region_filter != "any" and item.get("region") != region_filter:
                continue

            with st.expander(f"▶ {item.get('title','(untitled)')} — {item.get('region')} [{item.get('lang')}]"):
                st.caption(f"{item.get('ts','')} | Source: {item.get('source','')} | URL: {item.get('orig_url','')}")
                st.video(item["file"])
                st.markdown(f"**Summary**: {item.get('summary','')}")
                try:
                    with open(item["file"], "rb") as fh:
                        st.download_button(
                            "Download",
                            data=fh,
                            file_name=os.path.basename(item["file"]),
                            mime="video/mp4",
                            key=f"dl_arch_{os.path.basename(item['file'])}",
                        )
                except Exception:
                    pass
            shown += 1
            if shown >= show_n:
                break
