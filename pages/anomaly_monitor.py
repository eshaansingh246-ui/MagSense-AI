import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_generator import get_articles
from utils.models import run_isolation_forest, compute_dna, run_lda

COLORS = ["#c8102e", "#16a34a", "#d4a017", "#2563eb", "#9333ea"]


def show():
    st.markdown("""
    <div style='border-bottom: 3px solid #0f0e0a; padding-bottom: 1rem; margin-bottom: 1.5rem;'>
        <div class='sub-header'>Isolation Forest · Performance Monitoring</div>
        <div class='main-header'>Anomaly Monitor</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Running Isolation Forest…"):
        articles = get_articles()
        articles, _, _ = run_lda(articles)
        articles = run_isolation_forest(articles, contamination=0.08)
        articles = compute_dna(articles)

    anomalies = articles[articles["anomaly"]]
    spikes    = anomalies[anomalies["anomaly_type"] == "🚀 Spike"]
    flops     = anomalies[anomalies["anomaly_type"] == "⚠️ Flop"]

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Anomalies", len(anomalies), f"{len(anomalies)/len(articles)*100:.1f}% of corpus")
    k2.metric("🚀 Viral Spikes", len(spikes))
    k3.metric("⚠️ Content Flops", len(flops))
    k4.metric("Avg Engagement (Normal)", f"{articles[~articles['anomaly']]['engagement_score'].mean():.0f}")

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

    # ── Alert feed ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🚀 Viral Spikes")
        st.caption("These articles far outperformed expectations — understand why and replicate.")
        if spikes.empty:
            st.info("No spikes detected.")
        else:
            for _, row in spikes.sort_values("anomaly_score").head(8).iterrows():
                st.markdown(f"""
                <div style='background:#f0fdf4; border:1.5px solid #bbf7d0;
                            border-left:5px solid #16a34a; border-radius:4px;
                            padding:0.8rem 1rem; margin-bottom:0.6rem;'>
                    <div style='font-weight:700; font-size:0.9rem;'>{row['title']}</div>
                    <div style='font-size:0.77rem; color:#6b6560; margin-top:3px;'>
                        {row['category']} · Reads: <b>{row['reads']:,}</b> ·
                        Eng: <b>{row['engagement_score']}</b> ·
                        Shares: {row['shares']} · DNA: {row['dna_score']}
                    </div>
                    <div style='font-size:0.75rem; color:#16a34a; margin-top:3px; font-weight:600;'>
                        {row['anomaly_type']} — Anomaly Score: {row['anomaly_score']:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with col2:
        st.markdown("### ⚠️ Content Flops")
        st.caption("These articles severely underperformed — investigate and avoid similar content.")
        if flops.empty:
            st.info("No flops detected.")
        else:
            for _, row in flops.sort_values("anomaly_score", ascending=False).head(8).iterrows():
                st.markdown(f"""
                <div style='background:#fff5f5; border:1.5px solid #fecaca;
                            border-left:5px solid #c8102e; border-radius:4px;
                            padding:0.8rem 1rem; margin-bottom:0.6rem;'>
                    <div style='font-weight:700; font-size:0.9rem;'>{row['title']}</div>
                    <div style='font-size:0.77rem; color:#6b6560; margin-top:3px;'>
                        {row['category']} · Reads: <b>{row['reads']:,}</b> ·
                        Eng: <b>{row['engagement_score']}</b> ·
                        Shares: {row['shares']} · DNA: {row['dna_score']}
                    </div>
                    <div style='font-size:0.75rem; color:#c8102e; margin-top:3px; font-weight:600;'>
                        {row['anomaly_type']} — Anomaly Score: {row['anomaly_score']:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Anomaly scatter ───────────────────────────────────────────────────────
    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    st.markdown("### 📍 Engagement vs Reads — Anomaly Map")
    st.caption("Red = Anomaly · Grey = Normal. Hover for details.")

    articles["point_color"] = articles["anomaly_type"].map({
        "Normal": "#e8e3d8",
        "🚀 Spike": "#16a34a",
        "⚠️ Flop": "#c8102e",
    }).fillna("#e8e3d8")

    fig = px.scatter(
        articles, x="reads", y="engagement_score",
        color="anomaly_type",
        color_discrete_map={
            "Normal": "#e8e3d8",
            "🚀 Spike": "#16a34a",
            "⚠️ Flop": "#c8102e",
        },
        size="shares", size_max=18,
        hover_data=["title", "category", "dna_score", "anomaly_score"],
        opacity=0.75,
    )
    fig.update_layout(
        margin=dict(t=10, b=10), height=400,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#e8e3d8", title="Reads"),
        yaxis=dict(showgrid=True, gridcolor="#e8e3d8", title="Engagement Score"),
        legend_title="Type",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Anomaly by category ───────────────────────────────────────────────────
    st.markdown("### 📊 Anomaly Rate by Category")
    anom_rate = articles.groupby("category").agg(
        total=("article_id","count"),
        anomalies=("anomaly","sum")
    ).reset_index()
    anom_rate["rate_pct"] = (anom_rate["anomalies"] / anom_rate["total"] * 100).round(1)

    fig2 = px.bar(
        anom_rate.sort_values("rate_pct", ascending=False),
        x="category", y="rate_pct",
        color="rate_pct",
        color_continuous_scale=["#faf8f2", "#c8102e"],
        text_auto=".1f",
    )
    fig2.update_layout(
        margin=dict(t=10, b=10), height=280,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        yaxis=dict(showgrid=True, gridcolor="#e8e3d8", title="Anomaly Rate (%)"),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Raw anomaly table ─────────────────────────────────────────────────────
    with st.expander("📋 Full Anomaly Table"):
        show_cols = ["title","category","tone","reads","engagement_score",
                     "shares","dna_score","anomaly_type","anomaly_score"]
        st.dataframe(
            anomalies[show_cols].sort_values("anomaly_score").reset_index(drop=True),
            use_container_width=True,
        )
