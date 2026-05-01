"""
utils/data_generator.py
Generates realistic synthetic magazine data for MagSense AI demo.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Article corpus ────────────────────────────────────────────────────────────
TOPICS = {
    "Technology & AI": [
        "How AI is reshaping Indian newsrooms",
        "The rise of vernacular tech apps in Tier-2 cities",
        "5G rollout: What it means for digital publishers",
        "Generative AI tools every editor should know",
        "India's semiconductor ambitions: A deep dive",
        "Electric vehicles and the Indian middle class",
        "Fintech revolution in rural India",
        "Open-source AI models vs GPT: Who wins?",
    ],
    "Politics & Governance": [
        "Budget 2025: Winners and losers",
        "State election analysis: Shifting voter patterns",
        "New education policy: One year on",
        "Infrastructure spending and GDP growth",
        "Foreign policy shifts in the Modi era",
        "Urban local body reforms: Progress report",
    ],
    "Careers & Jobs": [
        "Top 10 careers of the decade",
        "How to negotiate a salary in 2025",
        "Remote work is reshaping Indian office culture",
        "Gig economy: Freedom or exploitation?",
        "MBA vs bootcamp: Which pays more?",
        "LinkedIn strategies that actually work",
        "Burnout is real—here's how to fight it",
    ],
    "Health & Wellness": [
        "Ayurveda meets modern medicine",
        "Mental health stigma in Indian workplaces",
        "Sleep science: Why Indians are chronically tired",
        "Gut microbiome and your mood",
        "Yoga's global expansion and its Indian roots",
    ],
    "Business & Finance": [
        "Startup funding winter: Survival strategies",
        "How D2C brands are eating FMCG's lunch",
        "Stock market volatility: A guide for retail investors",
        "Succession planning in family businesses",
        "Unicorns that flopped—and lessons learned",
    ],
    "Culture & Lifestyle": [
        "OTT vs cinema: The battle for Indian eyeballs",
        "Why Indian cooking is trending globally",
        "Street fashion in Mumbai: A photo essay",
        "The return of print magazines",
        "India's gaming generation",
    ],
}

TONES = ["Analytical", "Inspirational", "Critical", "Informative", "Conversational", "Investigative"]
STYLES = ["Long-form", "Listicle", "Q&A", "Opinion", "News", "Feature"]


def generate_articles(n=120) -> pd.DataFrame:
    rows = []
    today = datetime.today()
    for i in range(n):
        topic_cat = random.choice(list(TOPICS.keys()))
        title = random.choice(TOPICS[topic_cat])
        pub_date = today - timedelta(days=random.randint(0, 180))

        # Engagement varies by topic
        base_eng = {"Technology & AI": 780, "Politics & Governance": 520,
                    "Careers & Jobs": 710, "Health & Wellness": 630,
                    "Business & Finance": 660, "Culture & Lifestyle": 480}
        eng = max(50, int(np.random.normal(base_eng[topic_cat], 150)))
        reads = max(200, int(eng * random.uniform(1.5, 4.0)))
        time_spent = round(max(0.5, np.random.normal(3.8, 1.2)), 1)
        shares = max(0, int(eng * random.uniform(0.05, 0.25)))
        ctr = round(min(0.35, max(0.02, np.random.normal(0.12, 0.05))), 3)

        rows.append({
            "article_id": f"ART{1000+i}",
            "title": title,
            "category": topic_cat,
            "tone": random.choice(TONES),
            "style": random.choice(STYLES),
            "pub_date": pub_date.strftime("%Y-%m-%d"),
            "word_count": random.randint(400, 3200),
            "reads": reads,
            "engagement_score": eng,
            "time_spent_min": time_spent,
            "shares": shares,
            "ctr": ctr,
            "sentiment": round(random.uniform(-0.3, 0.9), 2),
        })
    return pd.DataFrame(rows)


def generate_readers(n=500) -> pd.DataFrame:
    segments = {
        "Tech Enthusiasts": 0.28,
        "Career Climbers": 0.22,
        "Finance Watchers": 0.18,
        "Health Seekers": 0.15,
        "Culture Explorers": 0.17,
    }
    seg_list = []
    for seg, prop in segments.items():
        seg_list.extend([seg] * int(n * prop))
    random.shuffle(seg_list)

    age_map = {
        "Tech Enthusiasts": (22, 35),
        "Career Climbers": (25, 40),
        "Finance Watchers": (30, 50),
        "Health Seekers": (28, 55),
        "Culture Explorers": (20, 45),
    }
    rows = []
    for i, seg in enumerate(seg_list[:n]):
        lo, hi = age_map[seg]
        rows.append({
            "reader_id": f"USR{5000+i}",
            "segment": seg,
            "age": random.randint(lo, hi),
            "sessions_per_week": round(max(0.5, np.random.normal(4.2, 1.8)), 1),
            "avg_time_per_session": round(max(1, np.random.normal(8.5, 3)), 1),
            "articles_read_monthly": random.randint(4, 40),
            "premium": random.random() < 0.32,
            "platform": random.choice(["Mobile", "Desktop", "Tablet"]),
            "city_tier": random.choice(["Tier 1", "Tier 1", "Tier 2", "Tier 2", "Tier 3"]),
        })
    return pd.DataFrame(rows)


def generate_trends() -> pd.DataFrame:
    trends = [
        {"topic": "AI Jobs & Automation", "momentum": 94, "volume_7d": 142000, "direction": "↑", "category": "Technology & AI"},
        {"topic": "Budget Impact on Startups", "momentum": 81, "volume_7d": 98000, "direction": "↑", "category": "Business & Finance"},
        {"topic": "Mental Health at Work", "momentum": 76, "volume_7d": 87000, "direction": "↑", "category": "Health & Wellness"},
        {"topic": "Electric Vehicles India", "momentum": 72, "volume_7d": 76000, "direction": "↑", "category": "Technology & AI"},
        {"topic": "OTT Content War", "momentum": 68, "volume_7d": 65000, "direction": "→", "category": "Culture & Lifestyle"},
        {"topic": "Gig Economy Rights", "momentum": 63, "volume_7d": 59000, "direction": "↑", "category": "Careers & Jobs"},
        {"topic": "Vernacular Language Apps", "momentum": 57, "volume_7d": 52000, "direction": "↑", "category": "Technology & AI"},
        {"topic": "Women in Leadership", "momentum": 54, "volume_7d": 48000, "direction": "→", "category": "Careers & Jobs"},
        {"topic": "Traditional Medicine Revival", "momentum": 49, "volume_7d": 41000, "direction": "↓", "category": "Health & Wellness"},
        {"topic": "Cryptocurrency Regulation", "momentum": 44, "volume_7d": 38000, "direction": "↓", "category": "Business & Finance"},
    ]
    return pd.DataFrame(trends)


def generate_time_series(days=90) -> pd.DataFrame:
    today = datetime.today()
    dates = [today - timedelta(days=d) for d in range(days, 0, -1)]
    cats = list(TOPICS.keys())
    rows = []
    for d in dates:
        for cat in cats:
            base = {"Technology & AI": 4200, "Politics & Governance": 2800,
                    "Careers & Jobs": 3800, "Health & Wellness": 3100,
                    "Business & Finance": 3400, "Culture & Lifestyle": 2500}
            val = max(100, int(np.random.normal(base[cat], 400)))
            rows.append({"date": d.strftime("%Y-%m-%d"), "category": cat, "reads": val})
    return pd.DataFrame(rows)


# ── Cached loaders ────────────────────────────────────────────────────────────
_articles = None
_readers = None
_trends = None
_timeseries = None


def get_articles():
    global _articles
    if _articles is None:
        _articles = generate_articles(120)
    return _articles


def get_readers():
    global _readers
    if _readers is None:
        _readers = generate_readers(500)
    return _readers


def get_trends():
    global _trends
    if _trends is None:
        _trends = generate_trends()
    return _trends


def get_timeseries():
    global _timeseries
    if _timeseries is None:
        _timeseries = generate_time_series(90)
    return _timeseries
