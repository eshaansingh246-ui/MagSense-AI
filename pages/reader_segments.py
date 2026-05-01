import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_generator import get_readers
from utils.models import run_kmeans

COLORS = ["#c8102e", "#2563eb", "#d4a017", "#16a34a", "#9333ea"]
PERSONA_ICONS = {
    "Tech Enthusiasts": "📱",
    "Career Climbers": "🚀",
    "Finance Watchers": "📊",
    "Health Seekers": "🌿",
    "Culture Explorers": "🎨",
}
PERSONA_DESC = {
    "Tech Enthusiasts": "Heavy mobile users, early adopters, skew 22–35. Engage most with AI, EV, and fintech content.",
    "Career Climbers": "Ambitious professionals, 25–40. Love salary guides, leadership advice, and productivity hacks.",
    "Finance Watchers": "Older, high-income readers, 30–50. Follow markets, budget analysis, and startup funding news.",
    "Health Seekers": "Wellness-driven, 28–55. Prefer science-backed health, mental wellbeing, and nutrition content.",
    "Culture Explorers": "Creative and curious, 20–45. Enjoy OTT reviews, travel, food, and lifestyle storytelling.",
}
RECOMMEND_MAP = {
    "Tech Enthusiasts": ["AI & Automation Trends", "EV India Deep Dive", "Vernacular App Boom"],
    "Career Climbers": ["Top 10 Careers of 2025", "Salary Negotiation Guide", "LinkedIn Strategies"],
    "Finance Watchers": ["Startup Funding Winter", "Market Volatility Guide", "D2C vs FMCG"],
    "Health Seekers": ["Gut Microbiome & Mood", "Sleep Science India", "Mental Health at Work"],
    "Culture Explorers": ["OTT Content War", "Indian Street Fashion", "Global Food Trends"],
}


def show():
    st.markdown("""
    <div style='border-bottom: 3px solid #0f0e0a; padding-bottom: 1rem; margin-bottom: 1.5rem;'>
        <div class='sub-header'>K-Means Clustering · Audience Intelligence</div>
        <div class='main-header'>Reader Segments</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Running K-Means clustering…"):
        readers = get_readers()
        readers, stats = run_kmeans(readers)

    # ── Persona cards ─────────────────────────────────────────────────────────
    st.markdown("### 👥 Reader Personas (K = 5 Clusters)")
    persona_cols = st.columns(5)
    personas = stats["persona"].tolist()
    for i, row in stats.iterrows():
        p = row["persona"]
        icon = PERSONA_ICONS.get(p, "👤")
        color = COLORS[i % len(COLORS)]
        with persona_cols[i % 5]:
            st.markdown(f"""
            <div style='background:#faf8f2; border:1.5px solid #e8e3d8;
                        border-top:5px solid {color}; border-radius:4px;
                        padding:1rem 1rem; text-align:center; margin-bottom:1rem;'>
                <div style='font-size:2rem;'>{icon}</div>
                <div style='font-weight:700; font-size:0.88rem; margin:0.4rem 0;'>{p}</div>
                <div style='font-size:1.6rem; font-weight:900; color:{color};'>{row['count']}</div>
                <div style='font-size:0.72rem; color:#6b6560;'>readers</div>
                <div style='font-size:0.75rem; margin-top:0.5rem;'>
                    {row['avg_sessions']} sessions/wk<br>
                    {row['avg_articles']} articles/mo<br>
                    {row['premium_pct']}% premium
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

    # ── Deep dive on selected persona ─────────────────────────────────────────
    st.markdown("### 🔍 Persona Deep Dive")
    sel_persona = st.selectbox("Select a Reader Persona", personas)

    p_data = readers[readers["persona"] == sel_persona]
    p_color = COLORS[personas.index(sel_persona) % len(COLORS)]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div class='insight-box'>
            <div style='font-size:2.5rem;'>{PERSONA_ICONS.get(sel_persona,'👤')}</div>
            <div style='font-family:Playfair Display,serif; font-size:1.4rem; font-weight:700; margin:0.5rem 0;'>{sel_persona}</div>
            <div style='font-size:0.85rem; opacity:0.8; line-height:1.6;'>{PERSONA_DESC.get(sel_persona,'')}</div>
            <div style='margin-top:1rem; font-size:0.82rem; opacity:0.7;'>
                📍 Count: {len(p_data)}<br>
                💎 Premium: {p_data['premium'].mean()*100:.1f}%<br>
                📱 Top Platform: {p_data['platform'].mode()[0]}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**📌 Recommended Articles**")
        for rec in RECOMMEND_MAP.get(sel_persona, []):
            st.markdown(f"- {rec}")

    with col2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Age Distribution**")
            fig1 = px.histogram(p_data, x="age", nbins=15, color_discrete_sequence=[p_color])
            fig1.update_layout(margin=dict(t=10,b=10), height=200,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#e8e3d8"))
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            st.markdown("**Platform Split**")
            plat = p_data["platform"].value_counts().reset_index()
            plat.columns = ["platform", "count"]
            fig2 = px.pie(plat, values="count", names="platform",
                color_discrete_sequence=[p_color, "#e8e3d8", "#6b6560"], hole=0.5)
            fig2.update_layout(margin=dict(t=10,b=10), height=200,
                paper_bgcolor="rgba(0,0,0,0)", legend=dict(font=dict(size=10)))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Sessions / Week vs Articles / Month**")
        fig3 = px.scatter(p_data, x="sessions_per_week", y="articles_read_monthly",
                color="premium", color_discrete_map={True: p_color, False: "#e8e3d8"},
                opacity=0.7, size_max=6)
        fig3.update_layout(margin=dict(t=10,b=10), height=220,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#e8e3d8"),
            legend_title="Premium")
        st.plotly_chart(fig3, use_container_width=True)

    # ── City tier breakdown ────────────────────────────────────────────────────
    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    st.markdown("**🏙️ City Tier Distribution Across Segments**")
    tier_data = readers.groupby(["persona", "city_tier"]).size().reset_index(name="count")
    fig4 = px.bar(tier_data, x="persona", y="count", color="city_tier",
        color_discrete_sequence=COLORS, barmode="stack")
    fig4.update_layout(
        margin=dict(t=10,b=10), height=300,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#e8e3d8"),
        legend_title="City Tier",
    )
    st.plotly_chart(fig4, use_container_width=True)
