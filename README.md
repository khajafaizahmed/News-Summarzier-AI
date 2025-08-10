# 📰 Automated News to Video (Web App)

A shareable **Streamlit** app that:
- Picks random **Hard News** and **Soft News** from paywall-friendly feeds (Reuters/AP/NPR/BBC/Guardian).
- Scrapes with `newspaper3k` (with request timeout).
- Summarizes using a lightweight model (`sshleifer/distilbart-cnn-12-6`) for faster cloud performance.
- Converts the summary to speech with `gTTS`.
- Generates a video frame with on-screen captions via **Pillow** (no ImageMagick required).
- Streams results in a **3-column grid** with source attribution, article links, and timestamps.
- Lets viewers **download** each video.
- Includes **Clear output** and a **slider** to choose videos per category (1–3), plus an optional **seed** for reproducible picks.

---

## 🚀 Local Run
```bash
git clone https://github.com/YOUR_USERNAME/news-to-video-web.git
cd news-to-video-web
pip install -r requirements.txt
streamlit run app.py
```

> First run may take a few minutes (model download); subsequent runs are faster due to caching.

---

## 🌐 Deploy (Streamlit Community Cloud)

1. Push this folder to GitHub.
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud) → **New app** → select repo → `app.py` → **Deploy**.
3. Share the public URL.

---

## 📁 Output

Generated files are stored in `output/` on the server and streamed in the page.

---

## 📝 Notes / Tips

* **Feeds:** HTTPS & true RSS (incl. `https://feeds.apnews.com/apf-topnews`).
* **Performance:** Article text is capped (~4000 chars) before summarization.
* **Fonts:** For consistent rendering, add `assets/DejaVuSans.ttf` (optional).
* **TTS:** `gTTS` needs internet; brief failures are handled.

---

## 🔧 Tech

Python, Streamlit, Hugging Face `transformers`, `gTTS`, `moviepy`, `imageio-ffmpeg`, `pillow`, `feedparser`, `newspaper3k`, `nltk`.
