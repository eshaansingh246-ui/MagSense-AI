import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_generator import get_trends, get_timeseries, get_articles
from utils.models import score_trends

COLORS = ["#c8102e", "#0f0e0a", "#d4a017", "#2563eb", "#16a34a", "#9333ea"]


def show():
    st.markdown("""
    <div style='border-bottom: 3px solid #0f0e0a; padding-bottom: 1rem; margin-bottom: 1.5rem;'>
        <div class='sub-header'>Social Media + News Signal Detection</div>
        <div class='main-header'>Trend Radar</div>
    </div>
    """, unsafe_allow_html=True)

    trends  = score_trends(get_trends())
    ts      = get_timeseries()
    articles = get_articles()

    # ── Urgency banner ────────────────────────────────────────────────────────
    viral = trends[trends["urgency"] == "🔥 Viral"]
    if not viral.empty:
        for _, row in viral.iterrows():
            st.markdown(f"""
            <div style='background:#c8102e; color:white; padding:0.8rem 1.4rem;
                        border-radius:4px; margin-bottom:0.8rem; font-size:0.92rem;'>
                🔥 <b>VIRAL ALERT:</b> "{row['topic']}" is trending at <b>{row['momentum']}/100</b>
                momentum with <b>{row['volume_7d']:,}</b> weekly mentions — <b>publish NOW</b>
            </div>
            """, unsafe_allow_html=True)

    # ── Trend table ───────────────────────────────────────────────────────────
    st.markdown("### 📡 All Tracked Trends")

    col_filter = st.columns([1, 1, 2])
    with col_filter[0]:
        min_mom = st.slider("Min Momentum", 0, 100, 40)
    with col_filter[1]:
        cat_filter = st.selectbox("Category", ["All"] + sorted(trends["category"].unique().tolist()))

    filtered = trends[trends["momentum"] >= min_mom]
    if cat_filter != "All":
        filtered = filtered[filtered["category"] == cat_filter]

    for _, row in filtered.sort_values("momentum", ascending=False).iterrows():
        pct   = int(row["momentum"])
        color = "#c8102e" if pct >= 80 else "#d4a017" if pct >= 65 else "#16a34a"
        urg   = str(row["urgency"])
        st.markdown(f"""
        <div style='background:#faf8f2; border:1.5px solid #e8e3d8;
                    border-left:5px solid {color}; border-radius:4px;
                    padding:0.9rem 1.3rem; margin-bottom:0.6rem;'>
            <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                <div style='flex:1;'>
                    <div style='font-weight:700; font-size:1rem;'>{row['topic']} <span style='font-size:0.9rem;'>{row['direction']}</span></div>
                    <div style='font-size:0.8rem; color:#6b6560; margin-top:2px;'>
                        {row['category']} · {row['volume_7d']:,} mentions this week · Urgency: {urg}
                    </div>
                </div>
                <div style='text-align:right; min-width:80px;'>
                    <div style='font-size:1.8rem; font-weight:900; color:{color}; line-height:1;'>{pct}</div>
                    <div style='font-size:0.7rem; color:#6b6560;'>momentum</div>
                </div>
            </div>
            <div style='background:#e8e3d8; border-radius:2px; height:5px; margin-top:10px;'>
                <div style='background:{color}; width:{pct}%; height:5px; border-radius:2px;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Radar chart ───────────────────────────────────────────────────────────
    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    st.markdown("### 🕸️ Category Momentum Radar")

    cat_mom = trends.groupby("category")["momentum"].mean().reset_index()
    categories = cat_mom["category"].tolist() + [cat_mom["category"].tolist()[0]]
    values     = cat_mom["momentum"].tolist() + [cat_mom["momentum"].tolist()[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values, theta=categories,
        fill="toself",
        fillcolor="rgba(200,16,46,0.15)",
        line=dict(color="#c8102e", width=2),
        name="Avg Momentum",
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#e8e3d8"),
            angularaxis=dict(gridcolor="#e8e3d8"),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=20),
        height=380,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Volume bar chart ──────────────────────────────────────────────────────
    st.markdown("### 📊 Weekly Mention Volume by Topic")
    fig_vol = px.bar(
        trends.sort_values("volume_7d", ascending=True),
        x="volume_7d", y="topic", orientation="h",
        color="momentum", color_continuous_scale=["#faf8f2", "#c8102e"],
        text="volume_7d",
    )
    fig_vol.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_vol.update_layout(
        margin=dict(t=10, b=10), height=380,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        xaxis=dict(showgrid=True, gridcolor="#e8e3d8"),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig_vol, use_container_width=True)
