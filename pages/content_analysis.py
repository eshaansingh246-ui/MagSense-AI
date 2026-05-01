import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_generator import get_articles
from utils.models import run_lda, compute_dna

COLORS = ["#c8102e", "#0f0e0a", "#d4a017", "#2563eb", "#16a34a", "#9333ea"]


def show():
    st.markdown("""
    <div style='border-bottom: 3px solid #0f0e0a; padding-bottom: 1rem; margin-bottom: 1.5rem;'>
        <div class='sub-header'>NLP + LDA Analysis</div>
        <div class='main-header'>Content Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Running LDA topic modelling…"):
        articles = get_articles()
        articles, topic_keywords, doc_topics = run_lda(articles)
        articles = compute_dna(articles)

    # ── Topic keyword explorer ────────────────────────────────────────────────
    st.markdown("### 🧠 Discovered Content Topics (LDA)")
    st.caption("Latent Dirichlet Allocation identifies hidden thematic clusters across all articles.")

    topic_cols = st.columns(3)
    for i, (label, kws) in enumerate(topic_keywords.items()):
        with topic_cols[i % 3]:
            color = COLORS[i % len(COLORS)]
            kw_html = " · ".join(kws)
            st.markdown(f"""
            <div style='background:#faf8f2; border:1.5px solid #e8e3d8;
                        border-top:4px solid {color}; border-radius:4px;
                        padding:1rem 1.2rem; margin-bottom:1rem; min-height:110px;'>
                <div style='font-weight:700; font-size:0.95rem; margin-bottom:0.5rem;'>{label}</div>
                <div style='font-size:0.8rem; color:#6b6560; line-height:1.7;'>{kw_html}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

    # ── Filter + article table ────────────────────────────────────────────────
    st.markdown("### 📋 Article-Level Analysis")

    c1, c2, c3 = st.columns(3)
    with c1:
        sel_cat = st.selectbox("Filter by Category", ["All"] + sorted(articles["category"].unique().tolist()))
    with c2:
        sel_tone = st.selectbox("Filter by Tone", ["All"] + sorted(articles["tone"].unique().tolist()))
    with c3:
        sel_grade = st.selectbox("DNA Grade", ["All", "A+", "A", "B", "C", "D"])

    filtered = articles.copy()
    if sel_cat != "All":  filtered = filtered[filtered["category"] == sel_cat]
    if sel_tone != "All": filtered = filtered[filtered["tone"] == sel_tone]
    if sel_grade != "All": filtered = filtered[filtered["dna_grade"] == sel_grade]

    display_cols = ["title", "category", "lda_topic", "tone", "style",
                    "engagement_score", "reads", "dna_score", "dna_grade", "success_prediction"]
    st.dataframe(
        filtered[display_cols].sort_values("dna_score", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=380,
    )

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

    # ── Engagement by topic + tone heatmap ───────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Avg Engagement by LDA Topic**")
        eng_by_topic = articles.groupby("lda_topic")["engagement_score"].mean().reset_index()
        eng_by_topic.columns = ["topic", "avg_engagement"]
        eng_by_topic = eng_by_topic.sort_values("avg_engagement", ascending=True)
        fig = px.bar(eng_by_topic, x="avg_engagement", y="topic", orientation="h",
                     color="avg_engagement", color_continuous_scale=["#faf8f2", "#c8102e"])
        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10), height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            xaxis=dict(showgrid=True, gridcolor="#e8e3d8"),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Tone × Style Engagement Heatmap**")
        heat = articles.groupby(["tone", "style"])["engagement_score"].mean().reset_index()
        heat_pivot = heat.pivot(index="tone", columns="style", values="engagement_score").fillna(0)
        fig2 = px.imshow(
            heat_pivot, color_continuous_scale=["#faf8f2", "#c8102e"],
            aspect="auto", text_auto=".0f",
        )
        fig2.update_layout(
            margin=dict(t=10, b=10, l=10, r=10), height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── BERT-style sentiment analysis (simulated) ─────────────────────────────
    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    st.markdown("### 🤗 BERT Sentiment Analysis")
    st.caption("Simulated BERT-based sentiment scores across article corpus.")

    sent_by_cat = articles.groupby("category")["sentiment"].mean().reset_index()
    sent_by_cat.columns = ["category", "avg_sentiment"]
    sent_by_cat["color"] = sent_by_cat["avg_sentiment"].apply(
        lambda x: "#16a34a" if x > 0.3 else "#d4a017" if x > 0 else "#c8102e"
    )
    fig3 = px.bar(
        sent_by_cat.sort_values("avg_sentiment"), x="category", y="avg_sentiment",
        color="avg_sentiment", color_continuous_scale=["#c8102e", "#faf8f2", "#16a34a"],
        range_color=[-0.5, 0.9], text_auto=".2f",
    )
    fig3.update_layout(
        margin=dict(t=10, b=10), height=280,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#e8e3d8", title="Sentiment Score"),
    )
    st.plotly_chart(fig3, use_container_width=True)
