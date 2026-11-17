import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
from urllib.parse import quote

# ============================================================
# FOOTIQ — FOOTBALL ANALYTICS & TACTICAL INTELLIGENCE
# Replace the existing dashboard.py with this file.
# ============================================================

st.set_page_config(
    page_title="FootIQ | Football Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(ROOT, "data", "gold")
RESULTS = os.path.join(ROOT, "results")
STREAMING = os.path.join(ROOT, "data", "streaming", "wyscout_events")

# ----------------------------- THEME -----------------------------

BG = "#0B0F17"
PANEL = "#151C28"
PANEL_2 = "#1C2635"
BORDER = "#2A3545"
TEXT = "#F1F4F8"
MUTED = "#A5B0C0"
GREEN = "#43C6B5"
LIME = "#F2B866"
GOLDEN = "#F28C6B"
RED = "#F06C85"
BLUE = "#69AEEB"
PURPLE = "#B39AF5"
PALETTE = [BLUE, GOLDEN, GREEN, PURPLE, RED, "#D5A6C8"]

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
    .stApp {{ background: {BG}; color: {TEXT}; }}
    [data-testid="stSidebar"] {{
        background: #111824;
        border-right: 1px solid {BORDER};
    }}
    [data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}
    .block-container {{ max-width: 1580px; padding-top: 1.5rem; padding-bottom: 3rem; }}
    h1, h2, h3 {{ color: {TEXT} !important; letter-spacing: -0.4px; }}
    p, label, span {{ }}
    [data-testid="stMetric"] {{
        background: linear-gradient(145deg, #1A2432, #131A25);
        border: 1px solid {BORDER};
        border-radius: 15px;
        padding: 17px 19px;
        min-height: 112px;
    }}
    [data-testid="stMetricLabel"] {{ color: {MUTED} !important; font-size: 0.82rem; }}
    [data-testid="stMetricValue"] {{ color: {TEXT} !important; font-weight: 800; }}
    [data-testid="stMetricDelta"] {{ font-size: 0.75rem; }}
    div[data-testid="stSelectbox"] label, div[data-testid="stTextInput"] label,
    div[data-testid="stMultiSelect"] label {{ color: {MUTED}; }}
    .hero {{
        background: radial-gradient(circle at 85% 15%, #35445E 0%, transparent 34%),
                    linear-gradient(115deg, #202D42, #151D2B 62%, #24283A);
        border: 1px solid #3A4960;
        border-radius: 20px;
        padding: 25px 30px;
        margin-bottom: 20px;
    }}
    .hero-kicker {{ color: {GOLDEN}; font-size: 0.72rem; font-weight: 800; letter-spacing: 2px; }}
    .hero-title {{ color: #F4FAF5; font-size: 2.15rem; font-weight: 800; line-height: 1.15; }}
    .hero-sub {{ color: #B7C2D2; font-size: 0.92rem; margin-top: 7px; }}
    .section-title {{ font-size: 1.18rem; font-weight: 800; color: {TEXT}; margin: 0 0 3px 0; }}
    .section-sub {{ color: {MUTED}; font-size: 0.8rem; margin-bottom: 12px; }}
    .eyebrow {{ color: {GOLDEN}; font-size: 0.68rem; font-weight: 800; letter-spacing: 1.5px; }}
    .club-chip {{
        display: inline-block; background: #27364B; color: #E5EDF8;
        border: 1px solid #465A75; border-radius: 8px;
        min-width: 34px; text-align: center; padding: 5px 7px;
        font-size: 0.72rem; font-weight: 800; margin-right: 7px;
    }}
    .note {{
        background: #192333; border: 1px solid {BORDER}; border-radius: 12px;
        padding: 12px 15px; color: {MUTED}; font-size: 0.82rem;
    }}
    div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 10px; }}
    .stButton > button {{
        background: #29384D; color: {TEXT}; border: 1px solid #526782;
        border-radius: 9px; font-weight: 700;
    }}
    .stButton > button:hover {{ background: #354761; color: white; border-color: {BLUE}; }}
    hr {{ border-color: {BORDER}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- DATA -----------------------------

@st.cache_data(show_spinner=False)
def load_parquet(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

match = load_parquet(os.path.join(GOLD, "match_summary_final"))
teams = load_parquet(os.path.join(GOLD, "team_statistics"))
players = load_parquet(os.path.join(GOLD, "player_statistics"))
shots = load_parquet(os.path.join(GOLD, "shot_analytics"))
edges = load_parquet(os.path.join(GOLD, "graph_pass_edges"))
vertices = load_parquet(os.path.join(GOLD, "graph_vertices"))
communities = load_parquet(os.path.join(GOLD, "player_communities"))
community_summary = load_parquet(os.path.join(GOLD, "community_edge_summary"))
clusters = load_parquet(os.path.join(GOLD, "player_clusters"))
streaming = load_parquet(STREAMING)
model_results = load_csv(os.path.join(RESULTS, "model_comparison.csv"))
benchmark = load_csv(os.path.join(RESULTS, "partition_benchmark.csv"))

# ----------------------------- HELPERS -----------------------------

def first_col(df, candidates):
    return next((c for c in candidates if c in df.columns), None)

def numeric_cols(df):
    return df.select_dtypes(include="number").columns.tolist() if not df.empty else []

def fmt_num(value):
    try:
        return f"{float(value):,.0f}"
    except Exception:
        return "—"

def clean_label(value):
    return str(value).replace("_", " ").title()

def plot_layout(fig, height=380, legend=True):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color=TEXT, size=11),
        height=height,
        margin=dict(l=12, r=12, t=45, b=12),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"
        ) if legend else dict(),
        hoverlabel=dict(bgcolor="#17291F", bordercolor=BORDER, font_color=TEXT),
    )
    fig.update_xaxes(gridcolor="#26372E", zerolinecolor="#26372E")
    fig.update_yaxes(gridcolor="#26372E", zerolinecolor="#26372E")
    return fig

def section(title, subtitle=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)

def safe_metric(df, col, agg="sum"):
    if df.empty or col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    return s.mean() if agg == "mean" else s.sum()

def club_initials(name):
    words = [w for w in str(name).replace("-", " ").split() if w]
    if not words:
        return "FC"
    if len(words) == 1:
        return words[0][:3].upper()
    return "".join(w[0] for w in words[:3]).upper()

COUNTRY_FLAGS = {
    "argentina": "🇦🇷", "australia": "🇦🇺", "austria": "🇦🇹",
    "belgium": "🇧🇪", "brazil": "🇧🇷", "cameroon": "🇨🇲",
    "canada": "🇨🇦", "chile": "🇨🇱", "china": "🇨🇳",
    "colombia": "🇨🇴", "croatia": "🇭🇷", "czech republic": "🇨🇿",
    "denmark": "🇩🇰", "egypt": "🇪🇬", "england": "🏴",
    "finland": "🇫🇮", "france": "🇫🇷", "germany": "🇩🇪",
    "ghana": "🇬🇭", "greece": "🇬🇷", "india": "🇮🇳",
    "iran": "🇮🇷", "ireland": "🇮🇪", "italy": "🇮🇹",
    "ivory coast": "🇨🇮", "japan": "🇯🇵", "mexico": "🇲🇽",
    "morocco": "🇲🇦", "netherlands": "🇳🇱", "nigeria": "🇳🇬",
    "norway": "🇳🇴", "poland": "🇵🇱", "portugal": "🇵🇹",
    "romania": "🇷🇴", "russia": "🇷🇺", "saudi arabia": "🇸🇦",
    "scotland": "🏴", "senegal": "🇸🇳", "serbia": "🇷🇸",
    "south korea": "🇰🇷", "spain": "🇪🇸", "sweden": "🇸🇪",
    "switzerland": "🇨🇭", "turkey": "🇹🇷", "ukraine": "🇺🇦",
    "united states": "🇺🇸", "uruguay": "🇺🇾", "wales": "🏴",
}


# Curated Wikipedia page titles for common clubs. Other teams try their team name.
CLUB_WIKI_TITLES = {
    "barcelona": "FC Barcelona",
    "real madrid": "Real Madrid CF",
    "atletico madrid": "Atlético Madrid",
    "atletico de madrid": "Atlético Madrid",
    "manchester united": "Manchester United F.C.",
    "manchester city": "Manchester City F.C.",
    "liverpool": "Liverpool F.C.",
    "arsenal": "Arsenal F.C.",
    "chelsea": "Chelsea F.C.",
    "tottenham": "Tottenham Hotspur F.C.",
    " juventus": "Juventus F.C.",
    "juventus": "Juventus F.C.",
    "inter": "Inter Milan",
    "internazionale": "Inter Milan",
    "ac milan": "AC Milan",
    "milan": "AC Milan",
    "roma": "AS Roma",
    "napoli": "SSC Napoli",
    "bayern munich": "FC Bayern Munich",
    "borussia dortmund": "Borussia Dortmund",
    "paris saint-germain": "Paris Saint-Germain F.C.",
    "psg": "Paris Saint-Germain F.C.",
    "ajax": "AFC Ajax",
    "benfica": "S.L. Benfica",
    "porto": "FC Porto",
    "sporting cp": "Sporting CP",
    "valencia": "Valencia CF",
    "sevilla": "Sevilla FC",
    "villarreal": "Villarreal CF",
    "athletic club": "Athletic Bilbao",
    "napoli": "SSC Napoli",
}

@st.cache_data(ttl=86400, show_spinner=False)
def get_club_logo(team_name):
    """Fetch a club page thumbnail from Wikimedia; return None if unavailable."""
    if not team_name:
        return None
    key = str(team_name).strip().lower()
    title = CLUB_WIKI_TITLES.get(key, str(team_name).strip())
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(title.replace(" ", "_"))
        response = requests.get(
            url,
            headers={"User-Agent": "FootIQ-Dashboard/1.0 (football analytics student project)"},
            timeout=4,
        )
        if response.ok:
            payload = response.json()
            image = payload.get("thumbnail", {}).get("source")
            if image:
                return image
    except Exception:
        return None
    return None

def show_club_identity(team_name, caption="Selected team", team_type=None):
    # National teams are represented with flags; clubs use crest thumbnails.
    is_national = (
        (team_type is not None and "national" in str(team_type).lower())
        or str(team_name).strip().lower() in COUNTRY_FLAGS
    )
    logo_url = None if is_national else get_club_logo(team_name)
    col_logo, col_name = st.columns([0.12, 0.88], vertical_alignment="center")
    with col_logo:
        if is_national:
            st.markdown(
                f'<div style="font-size:2.4rem;line-height:1.4;">{flag_for(team_name)}</div>',
                unsafe_allow_html=True,
            )
        elif logo_url:
            st.image(logo_url, width=66)
        else:
            st.markdown(
                f'<div class="club-chip" style="font-size:1rem;padding:13px 8px;">{club_initials(team_name)}</div>',
                unsafe_allow_html=True,
            )
    with col_name:
        st.markdown(f"### {team_name}")
        st.caption(caption)

def flag_for(country):
    if pd.isna(country):
        return "🌍"
    key = str(country).strip().lower()
    return COUNTRY_FLAGS.get(key, "🌍")

def render_table(df, height=360):
    if df.empty:
        st.info("No records are available for this selection.")
    else:
        st.dataframe(df, use_container_width=True, height=height, hide_index=True)

def pitch_figure():
    fig = go.Figure()
    line = "#D7E7DA"
    # 0-100 Wyscout-style coordinate pitch.
    shapes = [
        dict(type="rect", x0=0, y0=0, x1=100, y1=100, line=dict(color=line, width=2)),
        dict(type="line", x0=50, y0=0, x1=50, y1=100, line=dict(color=line, width=1)),
        dict(type="circle", x0=39, y0=39, x1=61, y1=61, line=dict(color=line, width=1)),
        dict(type="rect", x0=0, y0=21, x1=16, y1=79, line=dict(color=line, width=1)),
        dict(type="rect", x0=84, y0=21, x1=100, y1=79, line=dict(color=line, width=1)),
        dict(type="rect", x0=0, y0=36, x1=5, y1=64, line=dict(color=line, width=1)),
        dict(type="rect", x0=95, y0=36, x1=100, y1=64, line=dict(color=line, width=1)),
    ]
    fig.update_layout(shapes=shapes)
    fig.update_xaxes(range=[-2, 102], visible=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-2, 102], visible=False)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#173B2A", height=500,
        margin=dict(l=5, r=5, t=5, b=5),
        showlegend=True, font=dict(color=TEXT),
    )
    return fig

# ----------------------------- HERO + NAV -----------------------------

with st.sidebar:
    st.markdown("## ⚽ FOOTIQ")
    st.caption("FOOTBALL INTELLIGENCE PLATFORM")
    st.divider()
    page = st.radio(
        "EXPLORE",
        [
            "📊 Overview",
            "🏆 Team Explorer",
            "👤 Player Explorer",
            "🔗 Passing Network",
            "🎯 Shot Analytics",
            "🧩 Player Clustering",
            "⚡ Streaming & Benchmark",
        ],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**DATA LAYER STATUS**")
    for label, frame in [
        ("Matches", match), ("Teams", teams), ("Players", players),
        ("Shots", shots), ("Pass graph", edges), ("Clusters", clusters),
        ("Streaming", streaming),
    ]:
        st.write(f"{'🟢' if not frame.empty else '⚪'} {label}")
    st.divider()
    st.caption("Spark · GraphX · MLlib · Kafka · Plotly")

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">DATA-DRIVEN FOOTBALL INTELLIGENCE</div>
      <div class="hero-title">⚽ FootIQ Analytics</div>
      <div class="hero-sub">From match events to tactical patterns — explore teams, players,
      passing structures and shot outcomes.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Global KPIs
goal_col = first_col(shots, ["is_goal", "goal", "isGoal"])
goals = safe_metric(shots, goal_col) if goal_col else None
shot_goal_rate = (goals / len(shots) * 100) if goals is not None and len(shots) else None

kpis = st.columns(5)
kpi_values = [
    ("MATCHES", fmt_num(len(match)), "Processed match records"),
    ("TEAMS", fmt_num(len(teams)), "Team profiles"),
    ("PLAYERS", fmt_num(len(players)), "Player profiles"),
    ("SHOTS", fmt_num(len(shots)), "Shot events analysed"),
    ("PASS LINKS", fmt_num(len(edges)), "Directed pass connections"),
]
for col, (label, value, help_text) in zip(kpis, kpi_values):
    col.metric(label, value, help=help_text)

if shot_goal_rate is not None:
    st.caption(f"⚽ {fmt_num(goals)} recorded goals · {shot_goal_rate:.2f}% observed shot goal rate")

st.divider()

# ============================================================
# OVERVIEW
# ============================================================

if page == "📊 Overview":
    section("Competition snapshot", "A high-level view of the processed Wyscout data.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Graph vertices", fmt_num(len(vertices)))
    c2.metric("Community assignments", fmt_num(len(communities)))
    c3.metric("Streaming events", fmt_num(len(streaming)))
    c4.metric("Model variants", fmt_num(len(model_results)))

    left, right = st.columns([1.15, 1])
    with left:
        section("Data footprint", "Records available in each analytical layer.")
        volume = pd.DataFrame({
            "Dataset": ["Matches", "Teams", "Players", "Shots", "Pass links",
                        "Graph vertices", "Communities", "Stream events"],
            "Records": [len(match), len(teams), len(players), len(shots),
                        len(edges), len(vertices), len(communities), len(streaming)],
        })
        fig = px.bar(volume, x="Records", y="Dataset", orientation="h",
                     color="Records", color_continuous_scale=["#244A35", GREEN],
                     text="Records")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_coloraxes(showscale=False)
        plot_layout(fig, 390, False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        section("Project pipeline", "Technology components used in FootIQ.")
        pipeline = [
            ("01", "Data engineering", "Bronze → Silver → Gold"),
            ("02", "Graph analytics", "GraphX passing network"),
            ("03", "Machine learning", "Shot classification + K-Means"),
            ("04", "Stream processing", "Kafka + Spark Structured Streaming"),
            ("05", "Visualization", "Streamlit + Plotly"),
        ]
        for n, title, desc in pipeline:
            st.markdown(
                f"""<div style="display:flex;gap:13px;align-items:center;
                background:{PANEL};border:1px solid {BORDER};padding:12px 14px;
                border-radius:11px;margin-bottom:9px;">
                <div style="font-size:0.78rem;color:{LIME};font-weight:800;">{n}</div>
                <div><div style="font-weight:700;color:{TEXT};">{title}</div>
                <div style="font-size:0.75rem;color:{MUTED};">{desc}</div></div></div>""",
                unsafe_allow_html=True,
            )

    st.divider()
    left, right = st.columns(2)
    with left:
        section("Match status distribution")
        status_col = first_col(match, ["status", "match_status"])
        if status_col:
            status_data = match[status_col].fillna("Unknown").value_counts().reset_index()
            status_data.columns = ["Status", "Matches"]
            fig = px.pie(status_data, names="Status", values="Matches", hole=0.62,
                         color_discrete_sequence=PALETTE)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            plot_layout(fig, 330)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No match status column was found in match_summary_final.")

    with right:
        section("Shot outcome snapshot")
        if goal_col:
            outcome = pd.DataFrame({
                "Outcome": ["Goals", "Non-goals"],
                "Shots": [int(goals), max(0, len(shots) - int(goals))],
            })
            fig = px.pie(outcome, names="Outcome", values="Shots", hole=0.62,
                         color="Outcome", color_discrete_map={"Goals": GREEN, "Non-goals": "#52665A"})
            fig.update_traces(textinfo="percent+label")
            plot_layout(fig, 330)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("A goal indicator was not detected in shot_analytics.")

    section("Latest match records")
    render_table(match.head(12))

# ============================================================
# TEAM EXPLORER
# ============================================================

elif page == "🏆 Team Explorer":
    section("Team Explorer", "Compare team profiles and inspect performance indicators. Club crests are fetched from Wikimedia when available; initials are used if no image is found.")

    if teams.empty:
        st.warning("team_statistics is not available.")
    else:
        team_col = first_col(teams, ["team_name", "team", "name"])
        team_id_col = first_col(teams, ["team_id"])
        numeric = [c for c in numeric_cols(teams) if c not in [team_id_col]]
        selected_metric = st.selectbox(
            "Rank teams by",
            numeric,
            index=numeric.index(first_col(teams, ["total_events", "pass_attempts",
                                                   "shots", "shot_attempts"]) or numeric[0])
            if numeric else 0,
            key="team_metric",
        ) if numeric else None

        if team_col and selected_metric:
            ranked = teams[[team_col, selected_metric]].copy()
            ranked[selected_metric] = pd.to_numeric(ranked[selected_metric], errors="coerce")
            ranked = ranked.dropna().nlargest(15, selected_metric)
            ranked["Club"] = ranked[team_col].map(club_initials)
            ranked["Team label"] = ranked["Club"] + "  " + ranked[team_col].astype(str)

            a, b = st.columns([1.25, 1])
            with a:
                section(f"Top teams · {clean_label(selected_metric)}")
                fig = px.bar(ranked.sort_values(selected_metric), x=selected_metric,
                             y="Team label", orientation="h", text=selected_metric,
                             color=selected_metric, color_continuous_scale=["#254A35", GREEN],
                             hover_data=[team_col])
                fig.update_traces(texttemplate="%{text:,.1f}", textposition="outside")
                fig.update_coloraxes(showscale=False)
                plot_layout(fig, 470, False)
                st.plotly_chart(fig, use_container_width=True)
            with b:
                section("Team performance distribution")
                if len(numeric) >= 2:
                    x_metric = st.selectbox("Horizontal axis", numeric, key="team_x")
                    y_metric = st.selectbox(
                        "Vertical axis", [c for c in numeric if c != x_metric],
                        key="team_y"
                    )
                    fig = px.scatter(
                        teams, x=x_metric, y=y_metric,
                        hover_name=team_col,
                        size=selected_metric if selected_metric != x_metric and selected_metric != y_metric else None,
                        color=selected_metric,
                        color_continuous_scale=["#254A35", GREEN],
                    )
                    plot_layout(fig, 470)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough numeric team metrics for a scatterplot.")

        st.divider()
        if team_col:
            choices = sorted(teams[team_col].dropna().astype(str).unique())
            chosen = st.selectbox("Inspect a club", choices, key="club_detail")
            row = teams[teams[team_col].astype(str) == chosen].iloc[0]
            show_club_identity(chosen, "National-team flag or club crest, when available", row.get("team_type"))
            detail = pd.DataFrame({"Metric": row.index, "Value": row.values})
            render_table(detail, 300)

        section("All team records")
        render_table(teams)

# ============================================================
# PLAYER EXPLORER
# ============================================================

elif page == "👤 Player Explorer":
    section("Player Explorer", "Find players, compare per-match output and inspect statistical profiles.")

    if players.empty:
        st.warning("player_statistics is not available.")
    else:
        player_col = first_col(players, ["player_name"])
        team_col = first_col(players, ["team_name"])
        position_col = first_col(players, ["position"])
        country_col = first_col(players, ["passport_country"])
        filtered = players.copy()

        f1, f2, f3 = st.columns(3)
        with f1:
            query = st.text_input("Search player", placeholder="Type a player name")
        with f2:
            if position_col:
                positions = ["All"] + sorted(filtered[position_col].dropna().astype(str).unique())
                selected_position = st.selectbox("Position", positions)
            else:
                selected_position = "All"
        with f3:
            if team_col:
                clubs = ["All"] + sorted(filtered[team_col].dropna().astype(str).unique())
                selected_club = st.selectbox("Club", clubs)
            else:
                selected_club = "All"

        if query and player_col:
            filtered = filtered[filtered[player_col].astype(str).str.contains(query, case=False, na=False)]
        if selected_position != "All" and position_col:
            filtered = filtered[filtered[position_col].astype(str) == selected_position]
        if selected_club != "All" and team_col:
            filtered = filtered[filtered[team_col].astype(str) == selected_club]

        st.metric("Players matching filters", fmt_num(len(filtered)))

        metric_candidates = [
            c for c in ["events_per_match", "passes_per_match", "shots_per_match",
                        "assists_per_match", "key_passes_per_match", "pass_accuracy_pct"]
            if c in filtered.columns
        ]
        if metric_candidates and player_col:
            metric = st.selectbox("Leaderboard metric", metric_candidates)
            top = filtered.copy()
            top[metric] = pd.to_numeric(top[metric], errors="coerce")
            top = top.dropna(subset=[metric]).nlargest(12, metric)
            if country_col:
                top["Flag"] = top[country_col].map(flag_for)
                top["Display"] = top["Flag"] + "  " + top[player_col].astype(str)
            else:
                top["Display"] = top[player_col].astype(str)

            left, right = st.columns([1.15, 1])
            with left:
                section(f"Top players · {clean_label(metric)}")
                fig = px.bar(top.sort_values(metric), x=metric, y="Display",
                             orientation="h", color=metric,
                             color_continuous_scale=["#254A35", GREEN],
                             hover_data=[player_col, team_col] if team_col else [player_col])
                fig.update_coloraxes(showscale=False)
                plot_layout(fig, 440, False)
                st.plotly_chart(fig, use_container_width=True)

            with right:
                section("Player profile radar")
                names = sorted(filtered[player_col].dropna().astype(str).unique())
                if names:
                    selected_player = st.selectbox("Select player", names, key="radar_player")
                    p = filtered[filtered[player_col].astype(str) == selected_player].iloc[0]
                    radar_cols = [c for c in [
                        "events_per_match", "passes_per_match", "shots_per_match",
                        "assists_per_match", "key_passes_per_match"
                    ] if c in filtered.columns]
                    if len(radar_cols) >= 3:
                        means = pd.to_numeric(players[radar_cols].stack(), errors="coerce")
                        # Normalize each metric to the dataset's 95th percentile for a readable radar.
                        values = []
                        for col in radar_cols:
                            cap = pd.to_numeric(players[col], errors="coerce").quantile(0.95)
                            raw = pd.to_numeric(pd.Series([p[col]]), errors="coerce").iloc[0]
                            values.append(min(100, max(0, (raw / cap * 100) if cap and pd.notna(raw) else 0)))
                        labels = [clean_label(c) for c in radar_cols]
                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(
                            r=values + [values[0]], theta=labels + [labels[0]],
                            fill="toself", name=selected_player,
                            line=dict(color=BLUE, width=3),
                            fillcolor="rgba(89,209,139,0.20)",
                        ))
                        fig.update_layout(
                            polar=dict(
                                bgcolor="rgba(0,0,0,0)",
                                radialaxis=dict(visible=True, range=[0, 100],
                                                gridcolor="#34483B", color=MUTED),
                                angularaxis=dict(gridcolor="#34483B", color=TEXT),
                            )
                        )
                        plot_layout(fig, 440)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Not enough per-match features for a radar profile.")

        if country_col and player_col:
            section("Nationality snapshot")
            nationality = filtered[country_col].fillna("Unknown").value_counts().head(12).reset_index()
            nationality.columns = ["Country", "Players"]
            nationality["Flag"] = nationality["Country"].map(flag_for)
            nationality["Country display"] = nationality["Flag"] + " " + nationality["Country"].astype(str)
            fig = px.bar(nationality.sort_values("Players"), x="Players", y="Country display",
                         orientation="h", color="Players",
                         color_continuous_scale=["#254A35", GOLDEN])
            fig.update_coloraxes(showscale=False)
            plot_layout(fig, 360, False)
            st.plotly_chart(fig, use_container_width=True)

        section("Player records")
        render_table(filtered)

# ============================================================
# PASSING NETWORK
# ============================================================

elif page == "🔗 Passing Network":
    section("Passing Network Explorer", "Explore player-to-player passing connections derived from the GraphX edge and vertex datasets.")

    if edges.empty or vertices.empty:
        st.warning("The pass graph or graph vertex dataset is unavailable.")
    else:
        required = {"src", "dst", "pass_count"}
        if not required.issubset(edges.columns) or "id" not in vertices.columns:
            st.warning("Expected graph fields were not found. Check src/dst/pass_count and vertex id.")
        else:
            filtered_edges = edges.copy()
            team_id_col = first_col(edges, ["team_id"])
            team_name_col = first_col(vertices, ["team_name"])
            vertex_team_id = first_col(vertices, ["team_id"])

            if team_id_col and vertex_team_id:
                team_options = edges[team_id_col].dropna().unique().tolist()
                selected_team_id = st.selectbox("Select team ID", ["All teams"] + sorted(team_options))
                if selected_team_id != "All teams":
                    filtered_edges = filtered_edges[filtered_edges[team_id_col] == selected_team_id]
                    selected_vertices = vertices[vertices[vertex_team_id] == selected_team_id].copy()
                else:
                    selected_vertices = vertices.copy()
            else:
                selected_vertices = vertices.copy()

            max_edges = st.slider("Connections to display", 10, 150, 55, 5)
            filtered_edges["pass_count"] = pd.to_numeric(filtered_edges["pass_count"], errors="coerce")
            filtered_edges = filtered_edges.dropna(subset=["pass_count"]).nlargest(max_edges, "pass_count")

            node_ids = set(filtered_edges["src"].tolist()) | set(filtered_edges["dst"].tolist())
            selected_vertices = selected_vertices[selected_vertices["id"].isin(node_ids)].drop_duplicates("id")

            if not selected_vertices.empty and not filtered_edges.empty:
                # Circular layout is deterministic and makes the graph readable without an extra dependency.
                ids = selected_vertices["id"].tolist()
                n = len(ids)
                positions = {
                    node_id: (math.cos(2 * math.pi * i / max(n, 1)),
                              math.sin(2 * math.pi * i / max(n, 1)))
                    for i, node_id in enumerate(ids)
                }
                name_col = first_col(selected_vertices, ["player_name"])
                names = {
                    row["id"]: str(row[name_col]) if name_col else str(row["id"])
                    for _, row in selected_vertices.iterrows()
                }
                edge_x, edge_y = [], []
                for _, e in filtered_edges.iterrows():
                    if e["src"] in positions and e["dst"] in positions:
                        x0, y0 = positions[e["src"]]
                        x1, y1 = positions[e["dst"]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])

                node_x = [positions[i][0] for i in ids]
                node_y = [positions[i][1] for i in ids]
                degree = {i: 0 for i in ids}
                for _, e in filtered_edges.iterrows():
                    if e["src"] in degree:
                        degree[e["src"]] += float(e["pass_count"])
                    if e["dst"] in degree:
                        degree[e["dst"]] += float(e["pass_count"])

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y, mode="lines",
                    line=dict(width=1, color="rgba(112,183,142,0.28)"),
                    hoverinfo="skip", name="Pass connections",
                ))
                fig.add_trace(go.Scatter(
                    x=node_x, y=node_y, mode="markers+text",
                    marker=dict(
                        size=[max(10, min(34, 9 + math.sqrt(degree[i]) * 0.8)) for i in ids],
                        color=[degree[i] for i in ids],
                        colorscale=[[0, "#28533A"], [1, GREEN]],
                        line=dict(color="#DDF5E4", width=1),
                        colorbar=dict(title="Pass activity"),
                    ),
                    text=[names[i] for i in ids],
                    textposition="top center",
                    textfont=dict(size=9, color=TEXT),
                    customdata=[[i, degree[i]] for i in ids],
                    hovertemplate="<b>%{text}</b><br>ID: %{customdata[0]}<br>Weighted activity: %{customdata[1]:.0f}<extra></extra>",
                    name="Players",
                ))
                fig.update_xaxes(visible=False)
                fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", height=650,
                    margin=dict(l=10, r=10, t=20, b=10),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Node size and colour represent weighted passing activity within the displayed edge subset. Layout is a circular visualization, not player coordinates on a pitch.")
            else:
                st.info("No matching nodes or edges were found for this selection.")

            st.divider()
            left, right = st.columns(2)
            with left:
                section("Highest-volume pass links")
                render_table(filtered_edges.sort_values("pass_count", ascending=False).head(20))
            with right:
                section("Community structure")
                if not community_summary.empty:
                    render_table(community_summary)
                else:
                    st.info("Community edge summary is unavailable.")

# ============================================================
# SHOT ANALYTICS
# ============================================================

elif page == "🎯 Shot Analytics":
    section("Shot Analytics Lab", "Inspect shot locations, outcomes and baseline model performance.")

    if shots.empty:
        st.warning("shot_analytics is not available.")
    else:
        goal_col = first_col(shots, ["is_goal", "goal", "isGoal"])
        goals = safe_metric(shots, goal_col) if goal_col else None
        rate = goals / len(shots) * 100 if goals is not None and len(shots) else None

        a, b, c, d = st.columns(4)
        a.metric("Shot attempts", fmt_num(len(shots)))
        b.metric("Goals", fmt_num(goals) if goals is not None else "—")
        c.metric("Non-goals", fmt_num(len(shots) - goals) if goals is not None else "—")
        d.metric("Goal rate", f"{rate:.2f}%" if rate is not None else "—")

        x_col = first_col(shots, ["start_x", "x", "shot_x", "pos_x"])
        y_col = first_col(shots, ["start_y", "y", "shot_y", "pos_y"])
        if x_col and y_col:
            section("Shot map", "Shot locations plotted using the available event coordinates.")
            plot_df = shots.copy()
            plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
            plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
            plot_df = plot_df.dropna(subset=[x_col, y_col])
            # If coordinates appear to use 0–100 Wyscout pitch units, plot directly.
            if not plot_df.empty and plot_df[x_col].between(0, 100).mean() > 0.9 and plot_df[y_col].between(0, 100).mean() > 0.9:
                fig = pitch_figure()
                if goal_col:
                    plot_df["_outcome"] = plot_df[goal_col].astype(str).str.lower().isin(["1", "true", "yes"]).map(
                        {True: "Goal", False: "No goal"}
                    )
                    colors = {"Goal": GREEN, "No goal": "#E6B65C"}
                    for outcome, colour in colors.items():
                        part = plot_df[plot_df["_outcome"] == outcome]
                        fig.add_trace(go.Scatter(
                            x=part[x_col], y=100 - part[y_col],
                            mode="markers", name=outcome,
                            marker=dict(size=7, color=colour, opacity=0.72,
                                        line=dict(width=0.5, color="#0B1510")),
                            customdata=part[[x_col, y_col]].values,
                            hovertemplate=f"{outcome}<br>X: %{{x:.1f}}<br>Y: %{{customdata[1]:.1f}}<extra></extra>",
                        ))
                else:
                    fig.add_trace(go.Scatter(
                        x=plot_df[x_col], y=100 - plot_df[y_col], mode="markers",
                        marker=dict(size=7, color=GREEN, opacity=0.65),
                        name="Shots",
                    ))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Shot coordinates were found but do not appear to use a 0–100 pitch coordinate scale; no pitch map was drawn.")
        else:
            st.info("No recognizable start_x/start_y shot coordinate columns were found, so the pitch map is unavailable.")

        left, right = st.columns(2)
        with left:
            section("Shot outcome distribution")
            if goal_col:
                outcome = pd.DataFrame({
                    "Outcome": ["Goals", "Non-goals"],
                    "Count": [int(goals), int(len(shots) - goals)],
                })
                fig = px.pie(outcome, names="Outcome", values="Count", hole=0.62,
                             color="Outcome",
                             color_discrete_map={"Goals": GREEN, "Non-goals": "#586A5E"})
                fig.update_traces(textinfo="label+percent")
                plot_layout(fig, 340)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Goal indicator column not found.")

        with right:
            section("Model comparison")
            if not model_results.empty:
                model_col = first_col(model_results, ["model", "Model", "algorithm"])
                metrics = [c for c in ["precision", "recall", "f1", "f1_score"] if c in model_results.columns]
                if model_col and metrics:
                    long = model_results.melt(
                        id_vars=[model_col], value_vars=metrics,
                        var_name="Metric", value_name="Score"
                    )
                    fig = px.bar(long, x=model_col, y="Score", color="Metric",
                                 barmode="group", text_auto=".3f",
                                 color_discrete_sequence=[GREEN, GOLDEN, BLUE, PURPLE])
                    fig.update_yaxes(range=[0, 1], title="Score")
                    plot_layout(fig, 340)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    render_table(model_results)
            else:
                st.info("Model comparison CSV is unavailable.")

        st.markdown(
            '<div class="note">Interpretation note: these are baseline shot-outcome classifiers using a limited feature set. They should not be described as a production expected-goals (xG) model.</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Inspect shot records"):
            render_table(shots.head(500))

# ============================================================
# PLAYER CLUSTERING
# ============================================================

elif page == "🧩 Player Clustering":
    section("Player Archetypes", "Explore the four K-Means groups derived from standardized per-match statistics.")

    if clusters.empty or "cluster" not in clusters.columns:
        st.warning("player_clusters is not available. Confirm the Gold Parquet output exists.")
    else:
        counts = clusters["cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Players")
        total = counts["Players"].sum()
        counts["Share"] = counts["Players"] / total * 100 if total else 0

        a, b = st.columns([1, 1.2])
        with a:
            section("Cluster distribution")
            fig = px.pie(counts, names="Cluster", values="Players", hole=0.66,
                         color_discrete_sequence=PALETTE)
            fig.update_traces(textinfo="label+percent", marker=dict(line=dict(color=BG, width=2)))
            plot_layout(fig, 380)
            st.plotly_chart(fig, use_container_width=True)
        with b:
            section("Cluster sizes")
            fig = px.bar(counts, x="Cluster", y="Players", text="Players",
                         color="Cluster", color_discrete_sequence=PALETTE)
            fig.update_traces(textposition="outside")
            plot_layout(fig, 380, False)
            st.plotly_chart(fig, use_container_width=True)

        feature_cols = [c for c in [
            "events_per_match", "passes_per_match", "shots_per_match",
            "assists_per_match", "key_passes_per_match"
        ] if c in clusters.columns]

        if feature_cols:
            section("Cluster statistical profiles", "Mean values are shown in their original feature units.")
            profiles = clusters.groupby("cluster")[feature_cols].mean().reset_index()
            long = profiles.melt(id_vars="cluster", var_name="Metric", value_name="Average")
            fig = px.line(long, x="Metric", y="Average", color="cluster",
                          markers=True, line_shape="spline",
                          color_discrete_sequence=PALETTE)
            fig.update_xaxes(tickangle=-20)
            plot_layout(fig, 430)
            st.plotly_chart(fig, use_container_width=True)

            # Radar uses per-feature min-max scaling across the four cluster means.
            radar = profiles.copy()
            for col in feature_cols:
                lo, hi = radar[col].min(), radar[col].max()
                radar[col] = ((radar[col] - lo) / (hi - lo) * 100) if hi != lo else 50
            fig = go.Figure()
            for _, row in radar.iterrows():
                vals = [row[c] for c in feature_cols]
                labels = [clean_label(c) for c in feature_cols]
                fig.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], theta=labels + [labels[0]],
                    fill="toself", name=f"Cluster {int(row['cluster'])}",
                    line=dict(width=2),
                ))
            fig.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="#34483B", color=MUTED),
                    angularaxis=dict(gridcolor="#34483B", color=TEXT),
                )
            )
            plot_layout(fig, 470)
            section("Relative cluster radar", "Each metric is scaled from 0–100 across the four cluster means; this compares profile shape, not absolute volume.")
            st.plotly_chart(fig, use_container_width=True)

        selected_cluster = st.selectbox("Explore cluster members", sorted(clusters["cluster"].dropna().unique()))
        members = clusters[clusters["cluster"] == selected_cluster].copy()
        st.metric("Players in selected cluster", fmt_num(len(members)))
        render_table(members)

        st.markdown(
            '<div class="note">Cluster labels are statistical group IDs, not verified football positions. Interpret them from the feature profiles rather than assigning tactical roles without additional evidence.</div>',
            unsafe_allow_html=True,
        )

# ============================================================
# STREAMING & BENCHMARK
# ============================================================

elif page == "⚡ Streaming & Benchmark":
    section("Streaming & Performance", "Historical Wyscout events replayed through a local Kafka topic and persisted by Spark Structured Streaming.")

    a, b, c = st.columns(3)
    a.metric("Persisted stream events", fmt_num(len(streaming)))
    b.metric("Pass graph edges", fmt_num(len(edges)))
    c.metric("Benchmark configurations", fmt_num(len(benchmark)))

    if not streaming.empty:
        event_col = first_col(streaming, ["event_name", "event_type"])
        left, right = st.columns([1, 1])
        with left:
            section("Event type distribution")
            if event_col:
                event_counts = streaming[event_col].fillna("Unknown").value_counts().head(15).reset_index()
                event_counts.columns = ["Event", "Count"]
                fig = px.bar(event_counts.sort_values("Count"), x="Count", y="Event",
                             orientation="h", color="Count",
                             color_continuous_scale=["#254A35", GREEN])
                fig.update_coloraxes(showscale=False)
                plot_layout(fig, 410, False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No event type column was found.")
        with right:
            section("Streaming schema")
            schema = pd.DataFrame({
                "Field": streaming.columns,
                "Data type": [str(t) for t in streaming.dtypes],
            })
            render_table(schema, 410)
        with st.expander("Preview persisted streaming events"):
            render_table(streaming.head(100))
    else:
        st.info("Streaming Parquet output was not found at data/streaming/wyscout_events.")

    st.divider()
    section("Spark partition benchmark", "Local microbenchmark of the team aggregation at different shuffle partition settings.")
    if not benchmark.empty:
        render_table(benchmark)
        if {"partitions", "median_seconds"}.issubset(benchmark.columns):
            bench = benchmark.copy()
            bench["partitions"] = pd.to_numeric(bench["partitions"], errors="coerce")
            bench["median_seconds"] = pd.to_numeric(bench["median_seconds"], errors="coerce")
            bench = bench.dropna(subset=["partitions", "median_seconds"]).sort_values("partitions")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=bench["partitions"], y=bench["median_seconds"],
                mode="lines+markers+text",
                text=bench["median_seconds"].map(lambda x: f"{x:.4f}s"),
                textposition="top center",
                line=dict(color=BLUE, width=3),
                marker=dict(size=11, color=GOLDEN, line=dict(color=TEXT, width=1)),
                name="Median runtime",
            ))
            fig.update_xaxes(title="Shuffle partitions", tickmode="array",
                             tickvals=bench["partitions"].tolist())
            fig.update_yaxes(title="Median execution time (seconds)")
            plot_layout(fig, 390, False)
            st.plotly_chart(fig, use_container_width=True)
            best = bench.loc[bench["median_seconds"].idxmin()]
            st.markdown(
                f'<div class="note">Fastest observed setting in this local test: '
                f'<b>{int(best["partitions"])} partitions</b> at '
                f'<b>{best["median_seconds"]:.4f}s</b>. This is a small local microbenchmark, '
                f'not a general cluster-tuning conclusion.</div>',
                unsafe_allow_html=True,
            )
    else:
        st.warning("results/partition_benchmark.csv is unavailable.")

# ----------------------------- FOOTER -----------------------------

st.divider()
st.markdown(
    f'<div style="text-align:center;color:{MUTED};font-size:0.75rem;">'
    'FOOTIQ · Distributed Football Analytics & Tactical Intelligence<br>'
    'Apache Spark · GraphX · PySpark MLlib · Apache Kafka · Streamlit · Plotly'
    '</div>',
    unsafe_allow_html=True,
)
