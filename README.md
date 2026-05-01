# 📰 MagSense AI — Magazine Intelligence & Content Strategy Engine

> A multi-model AI system that helps magazine companies understand **what to publish, for whom, and why** — automatically.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## 🧠 What It Does

MagSense AI analyses articles and reader behaviour using a suite of unsupervised ML models:

| Model | Purpose |
|---|---|
| **LDA (Latent Dirichlet Allocation)** | Discover hidden content topics across articles |
| **K-Means Clustering** | Segment readers into behavioural personas |
| **Isolation Forest** | Detect viral spikes and content flops |
| **Content DNA Scoring** | Predict article success before publishing |
| **Trend Detection** | Surface emerging topics before they go viral |

---

## 🚀 Features

- **📊 Dashboard** — KPIs, content distribution, 90-day engagement trends, AI editorial recommendations
- **🧠 Content Analysis** — LDA topic modelling, BERT sentiment analysis, tone × style heatmaps
- **👥 Reader Segments** — K-Means personas with city tier, platform, and premium breakdowns
- **📈 Trend Radar** — Momentum scoring, viral alerts, weekly mention volumes
- **🧬 Content DNA** — Per-article success scores (0–100) + interactive "predict new article" tool
- **⚠️ Anomaly Monitor** — Isolation Forest flags spikes and flops with scatter visualisation

---

## 🗂️ Project Structure

```
magsense-ai/
├── app.py                  # Main Streamlit entry point
├── requirements.txt
├── pages/
│   ├── dashboard.py
│   ├── content_analysis.py
│   ├── reader_segments.py
│   ├── trend_radar.py
│   ├── content_dna.py
│   └── anomaly_monitor.py
└── utils/
    ├── data_generator.py   # Synthetic magazine data
    └── models.py           # LDA, K-Means, Isolation Forest, DNA scoring
```

---

## ⚡ Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/magsense-ai.git
cd magsense-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** — done in ~60 seconds ✅

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Streamlit** — UI framework
- **scikit-learn** — LDA, K-Means, Isolation Forest
- **Plotly** — Interactive charts
- **Pandas / NumPy** — Data processing

---

## 🇮🇳 Built For

Indian digital publishing platforms, regional language magazines, and media companies like Times Group. Designed to work without any external APIs — runs fully offline on synthetic data.

---

## 🏆 Hackathon Highlights

- ✅ Multi-model ML system (5 distinct models)
- ✅ Clear media industry use case
- ✅ Fully deployable in 48 hours
- ✅ Demo-ready: "This article was predicted to flop… and it did"

---

## 📄 License

MIT
