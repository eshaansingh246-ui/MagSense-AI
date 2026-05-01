import streamlit as st

st.set_page_config(
    page_title="MagSense AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --ink: #0f0e0a;
    --cream: #faf8f2;
    --accent: #c8102e;
    --gold: #d4a017;
    --muted: #6b6560;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main-header {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    color: #0f0e0a;
    letter-spacing: -1px;
    line-height: 1.1;
}

.sub-header {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #6b6560;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 500;
}

.metric-card {
    background: #faf8f2;
    border: 1.5px solid #e8e3d8;
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    border-left: 4px solid #c8102e;
}

.alert-card {
    background: #fff5f5;
    border: 1.5px solid #fecaca;
    border-radius: 4px;
    padding: 1rem 1.4rem;
    border-left: 4px solid #c8102e;
    margin-bottom: 0.7rem;
}

.trend-card {
    background: #f0fdf4;
    border: 1.5px solid #bbf7d0;
    border-radius: 4px;
    padding: 1rem 1.4rem;
    border-left: 4px solid #16a34a;
    margin-bottom: 0.7rem;
}

.insight-box {
    background: linear-gradient(135deg, #0f0e0a 0%, #2d2b25 100%);
    color: #faf8f2;
    border-radius: 6px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1rem;
}

.dna-badge {
    display: inline-block;
    background: #c8102e;
    color: white;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.section-rule {
    border: none;
    border-top: 2px solid #0f0e0a;
    margin: 1.5rem 0 0.5rem 0;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    letter-spacing: 0.05em;
}

div[data-testid="metric-container"] {
    background: #faf8f2;
    border: 1.5px solid #e8e3d8;
    border-radius: 4px;
    padding: 0.8rem 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <span style='font-family:Playfair Display,serif; font-size:1.8rem; font-weight:900; color:#0f0e0a;'>
            Mag<span style='color:#c8102e;'>Sense</span>
        </span>
        <div style='font-size:0.7rem; letter-spacing:0.2em; color:#6b6560; text-transform:uppercase; margin-top:2px;'>
            AI Editorial Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**Navigation**")
    page = st.radio(
        "",
        ["📊 Dashboard", "🧠 Content Analysis", "👥 Reader Segments",
         "📈 Trend Radar", "🧬 Content DNA", "⚠️ Anomaly Monitor"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("""
    <div style='font-size:0.75rem; color:#6b6560; line-height:1.6;'>
    <b>Models Active</b><br>
    🟢 BERT NLP<br>
    🟢 LDA Topic Model<br>
    🟢 K-Means Clustering<br>
    🟢 Trend Detector<br>
    🟢 Isolation Forest
    </div>
    """, unsafe_allow_html=True)

# ── Route pages ───────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    from pages.dashboard import show
    show()
elif page == "🧠 Content Analysis":
    from pages.content_analysis import show
    show()
elif page == "👥 Reader Segments":
    from pages.reader_segments import show
    show()
elif page == "📈 Trend Radar":
    from pages.trend_radar import show
    show()
elif page == "🧬 Content DNA":
    from pages.content_dna import show
    show()
elif page == "⚠️ Anomaly Monitor":
    from pages.anomaly_monitor import show
    show()
