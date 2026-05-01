"""
utils/models.py
Lightweight ML pipeline: LDA topic modelling, K-Means clustering,
Isolation Forest anomaly detection, and simple trend scoring.
All models are trained on synthetic data so the app runs with zero
external API calls and no GPU requirement.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LDA Topic Modelling
# ─────────────────────────────────────────────────────────────────────────────

TOPIC_LABELS = [
    "🤖 AI & Tech Innovation",
    "🗳️ Politics & Policy",
    "💼 Careers & Growth",
    "❤️ Health & Wellbeing",
    "💰 Business & Finance",
    "🎭 Culture & Lifestyle",
]

def run_lda(articles_df: pd.DataFrame, n_topics: int = 6):
    """Fit LDA on article titles and return topic distributions."""
    corpus = articles_df["title"].tolist()
    vectorizer = CountVectorizer(max_df=0.95, min_df=1, stop_words="english", max_features=500)
    dtm = vectorizer.fit_transform(corpus)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=15,
        learning_method="online",
    )
    lda.fit(dtm)
    doc_topics = lda.transform(dtm)          # shape: (n_articles, n_topics)
    dominant = doc_topics.argmax(axis=1)

    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = {}
    for idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-8:][::-1]]
        label = TOPIC_LABELS[idx] if idx < len(TOPIC_LABELS) else f"Topic {idx}"
        topic_keywords[label] = top_words

    articles_df = articles_df.copy()
    articles_df["lda_topic_id"] = dominant
    articles_df["lda_topic"] = [TOPIC_LABELS[d] if d < len(TOPIC_LABELS) else f"Topic {d}" for d in dominant]
    articles_df["topic_confidence"] = doc_topics.max(axis=1).round(3)

    return articles_df, topic_keywords, doc_topics


# ─────────────────────────────────────────────────────────────────────────────
# 2. K-Means Reader Segmentation
# ─────────────────────────────────────────────────────────────────────────────

CLUSTER_PERSONAS = {
    0: ("Tech Enthusiasts", "📱", "#3b82f6"),
    1: ("Career Climbers",  "🚀", "#8b5cf6"),
    2: ("Finance Watchers", "📊", "#f59e0b"),
    3: ("Health Seekers",   "🌿", "#10b981"),
    4: ("Culture Explorers","🎨", "#ec4899"),
}

def run_kmeans(readers_df: pd.DataFrame, k: int = 5):
    """Cluster readers and attach persona labels."""
    features = readers_df[["sessions_per_week", "avg_time_per_session",
                            "articles_read_monthly", "age"]].copy()
    features["premium_num"] = readers_df["premium"].astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    readers_df = readers_df.copy()
    readers_df["cluster_id"] = labels
    readers_df["persona"]    = [CLUSTER_PERSONAS[l][0] for l in labels]
    readers_df["persona_icon"] = [CLUSTER_PERSONAS[l][1] for l in labels]
    readers_df["persona_color"] = [CLUSTER_PERSONAS[l][2] for l in labels]

    # Cluster stats
    stats = readers_df.groupby("persona").agg(
        count=("reader_id", "count"),
        avg_sessions=("sessions_per_week", "mean"),
        avg_articles=("articles_read_monthly", "mean"),
        premium_pct=("premium", "mean"),
    ).reset_index()
    stats["avg_sessions"] = stats["avg_sessions"].round(1)
    stats["avg_articles"] = stats["avg_articles"].round(1)
    stats["premium_pct"]  = (stats["premium_pct"] * 100).round(1)

    return readers_df, stats


# ─────────────────────────────────────────────────────────────────────────────
# 3. Isolation Forest – Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────────

def run_isolation_forest(articles_df: pd.DataFrame, contamination: float = 0.08):
    """Flag articles with anomalous engagement patterns."""
    features = articles_df[["reads", "engagement_score", "time_spent_min",
                             "shares", "ctr", "word_count"]].copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    preds   = iso.fit_predict(X)       # -1 = anomaly, 1 = normal
    scores  = iso.score_samples(X)     # more negative = more anomalous

    articles_df = articles_df.copy()
    articles_df["anomaly"]       = preds == -1
    articles_df["anomaly_score"] = scores.round(4)

    # Classify spike vs. flop
    mean_eng = articles_df["engagement_score"].mean()
    articles_df["anomaly_type"] = "Normal"
    mask = articles_df["anomaly"]
    articles_df.loc[mask & (articles_df["engagement_score"] > mean_eng), "anomaly_type"] = "🚀 Spike"
    articles_df.loc[mask & (articles_df["engagement_score"] <= mean_eng), "anomaly_type"] = "⚠️ Flop"

    return articles_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Content DNA Scoring
# ─────────────────────────────────────────────────────────────────────────────

DNA_WEIGHTS = {
    "reads": 0.25,
    "engagement_score": 0.30,
    "shares": 0.20,
    "time_spent_min": 0.15,
    "ctr": 0.10,
}

def compute_dna(articles_df: pd.DataFrame) -> pd.DataFrame:
    """Compute a 0–100 Content DNA score predicting article success."""
    df = articles_df.copy()
    scaler = StandardScaler()
    feature_cols = list(DNA_WEIGHTS.keys())
    scaled = scaler.fit_transform(df[feature_cols])
    weights = np.array(list(DNA_WEIGHTS.values()))
    raw_score = scaled @ weights
    # Normalise to 0–100
    mn, mx = raw_score.min(), raw_score.max()
    df["dna_score"] = ((raw_score - mn) / (mx - mn) * 100).round(1)

    # Grade
    def grade(s):
        if s >= 80: return "A+"
        if s >= 65: return "A"
        if s >= 50: return "B"
        if s >= 35: return "C"
        return "D"

    df["dna_grade"] = df["dna_score"].apply(grade)
    df["success_prediction"] = (df["dna_score"] >= 50).map({True: "✅ Predicted Hit", False: "❌ Predicted Flop"})
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Trend Momentum Scoring (simple)
# ─────────────────────────────────────────────────────────────────────────────

def score_trends(trends_df: pd.DataFrame) -> pd.DataFrame:
    """Add a normalised momentum_pct column."""
    df = trends_df.copy()
    mx = df["momentum"].max()
    df["momentum_pct"] = (df["momentum"] / mx * 100).round(1)
    df["urgency"] = pd.cut(
        df["momentum"],
        bins=[0, 50, 70, 85, 100],
        labels=["Low", "Medium", "High", "🔥 Viral"],
    )
    return df
