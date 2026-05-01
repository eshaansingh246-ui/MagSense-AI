import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_generator import get_articles
from utils.models import run_lda, compute_dna

COLORS = ["#c8102e", "#0f0e0a", "#d4a017", "#2563eb", "#16a34a", "#9333ea"]
GRADE_COLORS = {"A+": "#16a34a", "A": "#22c55e", "B": "#d4a017", "C": "#f97316", "D": "#c8102e"}


def show():
    st.markdown("""
    <div style='border-bottom: 3px solid #0f0e0a; padding-bottom: 1rem; margin-bottom: 1.5rem;'>
        <div class='sub-header'>Predictive Content Scoring</div>
        <div class='main-header'>Content DNA Mapping</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='insight-box'>
        <div style='font-size:1rem; font-weight:700; margin-bottom:0.4rem;'>🧬 What is Content DNA?</div>
        <div style='font-size:0.88rem; opacity:0.85; line-height:1.7;'>
        Every article receives a <b>DNA Score (0–100)</b> computed from a weighted blend of
        reads, engagement, shares, time-on-page, and CTR.
        The score predicts whether a future article with similar characteristics will
        <b>succeed or flop — before it's even published.</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Computing Content DNA…"):
        articles = get_articles()
        articles, _, _ = run_lda(articles)
        articles = compute_dna(articles)

    # ── Grade distribution ────────────────────────────────────────────────────
    st.markdown("### 🏅 DNA Grade Distribution")
    grade_counts = articles["dna_grade"].value_counts().reindex(["A+","A","B","C","D"]).fillna(0).reset_index()
    grade_counts.columns = ["grade", "count"]
    grade_counts["color"] = grade_counts["grade"].map(GRADE_COLORS)

    cols = st.columns(5)
    for i, row in grade_counts.iterrows():
        with cols[i]:
            st.markdown(f"""
            <div style='text-align:center; background:#faf8f2; border:1.5px solid #e8e3d8;
                        border-top:5px solid {row['color']}; border-radius:4px; padding:1rem;'>
                <div style='font-size:1.8rem; font-weight:900; color:{row['color']};'>{row['grade']}</div>
                <div style='font-size:1.5rem; font-weight:700;'>{int(row['count'])}</div>
                <div style='font-size:0.72rem; color:#6b6560;'>articles</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

    # ── Top and bottom performers ─────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏆 Top Performing Articles (A+)")
        top = articles[articles["dna_grade"] == "A+"].sort_values("dna_score", ascending=False).head(8)
        for _, row in top.iterrows():
            st.markdown(f"""
            <div style='background:#f0fdf4; border:1.5px solid #bbf7d0; border-left:4px solid #16a34a;
                        border-radius:4px; padding:0.7rem 1rem; margin-bottom:0.5rem;'>
                <div style='font-weight:600; font-size:0.88rem;'>{row['title']}</div>
                <div style='font-size:0.75rem; color:#6b6560; margin-top:2px;'>
                    DNA: <b style='color:#16a34a;'>{row['dna_score']}</b> ·
                    Reads: {row['reads']:,} · Eng: {row['engagement_score']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### ⚠️ Underperforming Articles (D)")
        bot = articles[articles["dna_grade"] == "D"].sort_values("dna_score").head(8)
        for _, row in bot.iterrows():
            st.markdown(f"""
            <div style='background:#fff5f5; border:1.5px solid #fecaca; border-left:4px solid #c8102e;
                        border-radius:4px; padding:0.7rem 1rem; margin-bottom:0.5rem;'>
                <div style='font-weight:600; font-size:0.88rem;'>{row['title']}</div>
                <div style='font-size:0.75rem; color:#6b6560; margin-top:2px;'>
                    DNA: <b style='color:#c8102e;'>{row['dna_score']}</b> ·
                    Reads: {row['reads']:,} · Eng: {row['engagement_score']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Scatter: DNA score vs engagement ─────────────────────────────────────
    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    st.markdown("### 📈 DNA Score vs Engagement Score")

    fig = px.scatter(
        articles, x="dna_score", y="engagement_score",
        color="dna_grade", color_discrete_map=GRADE_COLORS,
        size="reads", size_max=20,
        hover_data=["title", "category", "success_prediction"],
        opacity=0.75,
    )
    fig.update_layout(
        margin=dict(t=10, b=10), height=380,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#e8e3d8", title="DNA Score"),
        yaxis=dict(showgrid=True, gridcolor="#e8e3d8", title="Engagement Score"),
        legend_title="Grade",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── DNA by category ───────────────────────────────────────────────────────
    st.markdown("### 📊 Average DNA Score by Category")
    dna_cat = articles.groupby("category")["dna_score"].mean().reset_index().sort_values("dna_score", ascending=False)
    fig2 = px.bar(
        dna_cat, x="category", y="dna_score",
        color="dna_score", color_continuous_scale=["#c8102e", "#d4a017", "#16a34a"],
        text_auto=".1f",
    )
    fig2.update_layout(
        margin=dict(t=10, b=10), height=280,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#e8e3d8"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Predict new article ───────────────────────────────────────────────────
    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    st.markdown("### 🔮 Predict Success of a New Article")
    st.caption("Adjust the sliders to simulate a hypothetical article and see its predicted DNA score.")

    pc1, pc2 = st.columns(2)
    with pc1:
        est_reads      = st.slider("Estimated Reads",       500,  15000, 5000, 250)
        est_engagement = st.slider("Estimated Engagement",  100,   1200, 600,  25)
        est_shares     = st.slider("Estimated Shares",        0,    300,  80,   5)
    with pc2:
        est_time       = st.slider("Avg Time on Page (min)", 0.5,  12.0,  4.0, 0.5)
        est_ctr        = st.slider("CTR",                   0.01,  0.40,  0.12, 0.01)

    import numpy as np
    weights = np.array([0.25, 0.30, 0.20, 0.15, 0.10])
    means   = articles[["reads","engagement_score","shares","time_spent_min","ctr"]].mean().values
    stds    = articles[["reads","engagement_score","shares","time_spent_min","ctr"]].std().values
    user_v  = np.array([est_reads, est_engagement, est_shares, est_time, est_ctr])
    scaled  = (user_v - means) / stds
    raw     = float(np.dot(scaled, weights))
    mn = articles["dna_score"].min(); mx_s = articles["dna_score"].max()
    # Rough normalise relative to article distribution
    pct_rank = float(np.mean(articles["dna_score"] < raw * 30 + 50))
    pred_score = round(min(100, max(0, raw * 20 + 55)), 1)
    pred_grade = "A+" if pred_score>=80 else "A" if pred_score>=65 else "B" if pred_score>=50 else "C" if pred_score>=35 else "D"
    pred_label = "✅ Predicted HIT" if pred_score >= 50 else "❌ Predicted FLOP"
    pred_color = "#16a34a" if pred_score >= 50 else "#c8102e"

    st.markdown(f"""
    <div style='background:#faf8f2; border:2px solid {pred_color}; border-radius:6px;
                padding:1.5rem 2rem; text-align:center; margin-top:1rem;'>
        <div style='font-size:3rem; font-weight:900; color:{pred_color};'>{pred_score}</div>
        <div style='font-size:1.1rem; font-weight:700; color:{pred_color}; margin-top:0.3rem;'>
            Grade {pred_grade} · {pred_label}
        </div>
        <div style='font-size:0.82rem; color:#6b6560; margin-top:0.5rem;'>
            Based on weighted blend of reads, engagement, shares, time-on-page, and CTR signals.
        </div>
    </div>
    """, unsafe_allow_html=True)
