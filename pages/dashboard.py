import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_generator import get_articles, get_readers, get_trends, get_timeseries
from utils.models import run_lda, run_kmeans, run_isolation_forest, compute_dna, score_trends

COLORS = ["#c8102e", "#0f0e0a", "#d4a017", "#2563eb", "#16a34a", "#9333ea"]


def show():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='border-bottom: 3px solid #0f0e0a; padding-bottom: 1rem; margin-bottom: 1.5rem;'>
        <div class='sub-header'>Editorial Intelligence Platform</div>
        <div class='main-header'>MagSense AI Dashboard</div>
        <div style='font-size:0.9rem; color:#6b6560; margin-top:0.4rem;'>
            Real-time content intelligence · Powered by NLP + Clustering + Anomaly Detection
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load & process data ───────────────────────────────────────────────────
    with st.spinner("Running ML models…"):
        articles = get_articles()
        readers  = get_readers()
        trends   = score_trends(get_trends())
        ts       = get_timeseries()

        articles, _, _ = run_lda(articles)
        articles        = run_isolation_forest(articles)
        articles        = compute_dna(articles)
        readers, r_stats = run_kmeans(readers)

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📰 Articles Analysed", f"{len(articles):,}")
    k2.metric("👥 Readers Profiled", f"{len(readers):,}")
    k3.metric("📈 Trending Topics", f"{len(trends[trends['momentum']>60])}")
    k4.metric("⚠️ Anomalies Detected", f"{articles['anomaly'].sum()}")
    k5.metric("🧬 Avg DNA Score", f"{articles['dna_score'].mean():.1f}/100")

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

    # ── Row 1: category distribution + time series ────────────────────────────
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Content Distribution by Topic**")
        cat_counts = articles["category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        fig_pie = px.pie(
            cat_counts, values="count", names="category",
            color_discrete_sequence=COLORS,
            hole=0.45,
        )
        fig_pie.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(font=dict(size=11)),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown("**Reader Engagement Over 90 Days (by Category)**")
        ts_pivot = ts.groupby(["date", "category"])["reads"].mean().reset_index()
        fig_line = px.line(
            ts_pivot, x="date", y="reads", color="category",
            color_discrete_sequence=COLORS,
        )
        fig_line.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(font=dict(size=11), title=""),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#e8e3d8"),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # ── Row 2: Top trends + editorial alerts ──────────────────────────────────
    col3, col4 = st.columns([1, 1])

    with col3:
        st.markdown("**🔥 Top Trending Topics Right Now**")
        top_trends = trends.nlargest(5, "momentum")
        for _, row in top_trends.iterrows():
            pct = int(row["momentum"])
            color = "#c8102e" if pct >= 80 else "#d4a017" if pct >= 65 else "#16a34a"
            st.markdown(f"""
            <div class='trend-card'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                        <div style='font-weight:600; font-size:0.95rem;'>{row['topic']}</div>
                        <div style='font-size:0.75rem; color:#6b6560;'>{row['category']} · {row['volume_7d']:,} mentions/week</div>
                    </div>
                    <div style='font-size:1.4rem; font-weight:900; color:{color};'>{pct}</div>
                </div>
                <div style='background:#e8e3d8; border-radius:2px; height:4px; margin-top:8px;'>
                    <div style='background:{color}; width:{pct}%; height:4px; border-radius:2px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col4:
        st.markdown("**💡 AI Editorial Recommendations**")
        # Compute over-published vs under-published
        cat_eng = articles.groupby("category")["engagement_score"].mean().sort_values(ascending=False)
        cat_count = articles["category"].value_counts()
        over_pub  = cat_count.idxmax()
        under_pub = cat_count.idxmin()
        top_eng   = cat_eng.idxmax()

        recs = [
            (f"You're over-publishing <b>{over_pub}</b> ({cat_count[over_pub]} articles) but engagement is only {int(cat_eng[over_pub])}/article.", "#c8102e"),
            (f"<b>{top_eng}</b> drives the highest engagement ({int(cat_eng[top_eng])}/article) — publish more.", "#16a34a"),
            (f"Topic '<b>{trends.iloc[0]['topic']}</b>' is trending at {trends.iloc[0]['momentum']} momentum — publish NOW.", "#2563eb"),
            (f"Under-served category: <b>{under_pub}</b> has only {cat_count[under_pub]} articles but loyal readers.", "#d4a017"),
        ]
        for txt, color in recs:
            st.markdown(f"""
            <div style='background:#faf8f2; border:1.5px solid #e8e3d8; border-left:4px solid {color};
                        border-radius:4px; padding:0.9rem 1.1rem; margin-bottom:0.6rem;
                        font-size:0.88rem; line-height:1.5;'>
                {txt}
            </div>
            """, unsafe_allow_html=True)

    # ── Row 3: Reader segment donut ───────────────────────────────────────────
    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    st.markdown("**👥 Reader Persona Breakdown**")
    seg_counts = readers["persona"].value_counts().reset_index()
    seg_counts.columns = ["persona", "count"]
    fig_seg = px.bar(
        seg_counts, x="persona", y="count",
        color="persona", color_discrete_sequence=COLORS,
        text="count",
    )
    fig_seg.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#e8e3d8"),
    )
    fig_seg.update_traces(textposition="outside")
    st.plotly_chart(fig_seg, use_container_width=True)
