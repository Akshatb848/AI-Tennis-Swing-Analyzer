"""
TennisIQ Analytics Dashboard — Streamlit-powered match review.

Views:
- Match Overview: scoreboard, key stats, rally breakdown, line call count
- Shot Analysis: placement heatmaps, swing type distribution
- Match Review: factual observations, corrections, line call history
- Trends: performance trends, win rate, consistency

Data Source:
- Real session data via DashboardDataProvider when available
- Fallback to sample data with "Demo Data" banner
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime

from tennis.dashboard.data_provider import (
    DashboardDataProvider,
    get_available_sessions,
)

st.set_page_config(
    page_title="TennisIQ",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0a0a0a; }
    .stat-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px; padding: 20px; margin: 8px 0;
        border: 1px solid #2a2a4a;
    }
    .score-display {
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        border-radius: 20px; padding: 30px; text-align: center;
        border: 2px solid #2a2a4a;
    }
    .score-display h1 { color: #00ff88; margin: 0; font-size: 3em; letter-spacing: 12px; }
    .score-display h3 { color: #c9d1d9; margin: 0 0 10px 0; }
    .section-header {
        color: #c9d1d9; font-size: 1.1em; font-weight: 600;
        border-bottom: 2px solid #00ff88; padding-bottom: 8px; margin: 20px 0 10px 0;
    }
    .demo-banner {
        background: linear-gradient(135deg, #1a1a2e, #2d1b69);
        border-radius: 12px; padding: 16px; margin-bottom: 20px;
        border: 1px solid #6c3bd4; text-align: center;
    }
    .demo-banner span { color: #c9d1d9; font-size: 0.95em; }
</style>
""", unsafe_allow_html=True)

# ── Sample Data Generator (fallback) ─────────────────────────────────────────
def generate_sample_data():
    """Generate realistic sample match data for demo mode."""
    np.random.seed(42)
    return {
        "match": {
            "player1": "Alex Chen", "player2": "Jordan Smith",
            "score": [[6, 3, 7], [4, 6, 5]], "duration": "2h 15m",
            "surface": "Hard Court", "match_type": "Singles",
        },
        "p1_stats": {
            "aces": 8, "double_faults": 2, "first_serve_pct": 67,
            "winners": 32, "unforced_errors": 22, "break_points_won": "4/7",
            "net_points_won": 8, "net_points_total": 12,
            "forehand_winners": 18, "backhand_winners": 10,
            "forehand_errors": 8, "backhand_errors": 12,
            "max_serve_speed": 131, "avg_serve_speed": 118,
        },
        "p2_stats": {
            "aces": 5, "double_faults": 4, "first_serve_pct": 61,
            "winners": 28, "unforced_errors": 30, "break_points_won": "3/6",
            "net_points_won": 5, "net_points_total": 10,
            "forehand_winners": 15, "backhand_winners": 9,
            "forehand_errors": 12, "backhand_errors": 14,
            "max_serve_speed": 125, "avg_serve_speed": 112,
        },
        "rally_breakdown": {
            "avg_length": 4.8, "max_length": 22,
            "distribution": {"1-3": 38, "4-6": 24, "7-9": 12, "10+": 6},
            "short_pct": 47.5, "medium_pct": 30.0, "long_pct": 22.5,
        },
        "line_calls": {
            "total": 18, "in": 12, "out": 6,
            "avg_confidence": 91.3,
            "challenges_made": 2, "challenges_successful": 1,
            "calls": [
                {"verdict": "OUT", "confidence": "95%", "distance": "4.2cm", "line": "Baseline", "point": 8},
                {"verdict": "IN", "confidence": "88%", "distance": "1.8cm", "line": "Sideline", "point": 15},
                {"verdict": "OUT", "confidence": "92%", "distance": "6.1cm", "line": "Service Line", "point": 23},
                {"verdict": "IN", "confidence": "72%", "distance": "0.9cm", "line": "Baseline", "point": 31},
                {"verdict": "OUT", "confidence": "97%", "distance": "12.3cm", "line": "Sideline", "point": 42},
            ],
        },
        "swing_distribution": {
            "Forehand": 45, "Backhand": 32, "Serve": 18, "Volley": 5,
        },
        "review": {
            "observations": [
                "Average rally length: 4.8 shots",
                "Longest rally: 22 shots",
                "47% of rallies ended within 3 shots",
                "Alex Chen: 55 of 80 points won (69%)",
                "Line calls: 18 total (6 out calls)",
            ],
            "p1_corrections": [
                "Reduce risk on non-attacking shots. Prioritize placement over power on returns.",
                "Extended rallies resulted in errors. Consider earlier shot selection to shorten points.",
                "Approach shots landing short allow passing shots. Deepen approach before moving forward.",
            ],
            "p2_corrections": [
                "Unforced errors (30) exceed winners (28). Reduce error rate on baseline exchanges.",
                "First serve percentage at 61%. Increase consistency with moderate-pace first serves.",
                "Court coverage at 38%. Wider split step and earlier preparation for lateral movement.",
            ],
            "drills": [
                "Wall Rally Consistency (15 min) — 50 forehands against a wall, consistent contact point",
                "Cross-Court Rally (15 min) — rally cross-court focusing on body rotation and weight transfer",
                "Serve Toss Alignment (10 min) — 30 serve tosses, toss at 1 o'clock position",
            ],
        },
    }


# ── Data Source Selection ────────────────────────────────────────────────────

provider = DashboardDataProvider()
is_demo_data = False
selected_session_id = None

# Check for available real sessions
available_sessions = get_available_sessions()

if available_sessions:
    # Show session selector in sidebar
    session_options = {
        f"{s['player1']} vs {s['player2']} ({s['surface']})": s['id']
        for s in available_sessions
    }
    session_options["Demo Data"] = "__demo__"
else:
    session_options = {"Demo Data": "__demo__"}

# Load data
data = None
if available_sessions:
    # Try to load real data
    selected_label = list(session_options.keys())[0]
    selected_session_id = session_options[selected_label]
    if selected_session_id != "__demo__":
        data = provider.get_match_data(selected_session_id)

if data is None:
    data = generate_sample_data()
    is_demo_data = True


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("TennisIQ")
st.sidebar.caption("Match Intelligence")

st.sidebar.markdown("---")

# Session selector
if len(session_options) > 1:
    st.sidebar.subheader("Select Match")
    selected_label = st.sidebar.selectbox(
        "Match",
        list(session_options.keys()),
        label_visibility="collapsed",
    )
    selected_session_id = session_options[selected_label]
    if selected_session_id == "__demo__":
        data = generate_sample_data()
        is_demo_data = True
    else:
        real_data = provider.get_match_data(selected_session_id)
        if real_data:
            data = real_data
            is_demo_data = False
        else:
            data = generate_sample_data()
            is_demo_data = True

st.sidebar.markdown("---")
st.sidebar.subheader("Dashboard View")
view = st.sidebar.radio(
    "View",
    ["Match Overview", "Shot Analysis", "Match Review", "Trends"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Current Match")
st.sidebar.text(f"{data['match']['player1']} vs {data['match']['player2']}")
st.sidebar.text(f"{data['match']['surface']}")
st.sidebar.text(f"{data['match']['duration']}")

# Demo data banner
if is_demo_data:
    st.markdown("""
    <div class="demo-banner">
        <span>📊 Demo Data — Record a match to see real results</span>
    </div>
    """, unsafe_allow_html=True)

# ── Match Overview ───────────────────────────────────────────────────────────
if "Match Overview" in view:
    st.markdown("## Match Overview")

    # Scoreboard
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.markdown(f"""
        <div class="score-display">
            <h3>{data['match']['player1']}</h3>
            <h1>{' '.join(str(s) for s in data['match']['score'][0])}</h1>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="text-align:center; padding:40px 0;">
            <span style="color:#00ff88; font-size:1.5em; font-weight:bold;">FINAL</span><br>
            <span style="color:#c9d1d9; font-size:2em;">vs</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="score-display">
            <h3>{data['match']['player2']}</h3>
            <h1>{' '.join(str(s) for s in data['match']['score'][1])}</h1>
        </div>
        """, unsafe_allow_html=True)

    # Key Stats Row
    st.markdown('<div class="section-header">Key Statistics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    stats_display = [
        ("Aces", data['p1_stats']['aces'], data['p2_stats']['aces']),
        ("Winners", data['p1_stats']['winners'], data['p2_stats']['winners']),
        ("UE", data['p1_stats']['unforced_errors'], data['p2_stats']['unforced_errors']),
        ("1st Serve %", f"{data['p1_stats']['first_serve_pct']}%", f"{data['p2_stats']['first_serve_pct']}%"),
        ("Max Speed", data['p1_stats']['max_serve_speed'], data['p2_stats']['max_serve_speed']),
        ("BP Won", data['p1_stats']['break_points_won'], data['p2_stats']['break_points_won']),
    ]
    for col, (label, v1, v2) in zip([c1, c2, c3, c4, c5, c6], stats_display):
        col.metric(label, v1, f"vs {v2}")

    # Player comparison radar
    st.markdown('<div class="section-header">Player Comparison</div>', unsafe_allow_html=True)
    categories = ['Serve', 'Return', 'Net Play', 'Accuracy', 'Power', 'Consistency']
    p1_vals = [0.82, 0.71, 0.67, 0.73, 0.85, 0.68]
    p2_vals = [0.68, 0.65, 0.50, 0.58, 0.72, 0.53]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=p1_vals + [p1_vals[0]], theta=categories + [categories[0]], fill='toself', name=data['match']['player1'], line_color='#00ff88', fillcolor='rgba(0,255,136,0.15)'))
    fig_radar.add_trace(go.Scatterpolar(r=p2_vals + [p2_vals[0]], theta=categories + [categories[0]], fill='toself', name=data['match']['player2'], line_color='#ff6b6b', fillcolor='rgba(255,107,107,0.15)'))
    fig_radar.update_layout(polar=dict(bgcolor='#0d1117', radialaxis=dict(visible=True, range=[0, 1], gridcolor='#2a2a4a')), paper_bgcolor='#0a0a0a', font_color='#c9d1d9', height=400, showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)

    # Rally Breakdown + Line Call Summary (side by side)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Rally Breakdown</div>', unsafe_allow_html=True)
        rb = data['rally_breakdown']
        st.metric("Average Rally Length", f"{rb['avg_length']} shots")
        st.metric("Longest Rally", f"{rb['max_length']} shots")

        fig_rally = go.Figure(data=[go.Bar(
            x=list(rb['distribution'].keys()),
            y=list(rb['distribution'].values()),
            marker_color=['#00ff88', '#4dff88', '#80ffaa', '#b3ffcc'],
        )])
        fig_rally.update_layout(
            paper_bgcolor='#0a0a0a', plot_bgcolor='#0d1117',
            font_color='#c9d1d9', height=250,
            xaxis_title="Shots per Rally", yaxis_title="Count",
        )
        st.plotly_chart(fig_rally, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Line Calls</div>', unsafe_allow_html=True)
        lc = data['line_calls']
        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("Total Calls", lc['total'])
        lc2.metric("Out Calls", lc['out'])
        lc3.metric("Avg Confidence", f"{lc['avg_confidence']}%")

        if lc['challenges_made'] > 0:
            st.caption(f"Challenges: {lc['challenges_successful']}/{lc['challenges_made']} successful")

        st.markdown('<div class="section-header">Serve Speed Distribution</div>', unsafe_allow_html=True)
        fig_speed = go.Figure()
        fig_speed.add_trace(go.Histogram(x=np.random.normal(data['p1_stats'].get('avg_serve_speed', 118), 8, 60), name=data['match']['player1'], marker_color='#00ff88', opacity=0.7))
        fig_speed.add_trace(go.Histogram(x=np.random.normal(data['p2_stats'].get('avg_serve_speed', 112), 10, 55), name=data['match']['player2'], marker_color='#ff6b6b', opacity=0.7))
        fig_speed.update_layout(paper_bgcolor='#0a0a0a', plot_bgcolor='#0d1117', font_color='#c9d1d9', height=250, xaxis_title="Speed (mph)", barmode='overlay')
        st.plotly_chart(fig_speed, use_container_width=True)

# ── Shot Analysis ────────────────────────────────────────────────────────────
elif "Shot Analysis" in view:
    st.markdown("## Shot Placement Analysis")

    # Court heatmap
    player = st.radio("Select Player", [data['match']['player1'], data['match']['player2']], horizontal=True)
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=np.random.rand(10, 6) * (0.8 if player == data['match']['player1'] else 0.6),
        colorscale='Viridis', showscale=True,
    ))
    fig_heatmap.update_layout(
        title="Shot Placement Heatmap",
        paper_bgcolor='#0a0a0a', plot_bgcolor='#0d1117', font_color='#c9d1d9', height=400,
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Swing Type Distribution")
        sd = data['swing_distribution']
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(sd.keys()), values=list(sd.values()),
            hole=0.4, marker=dict(colors=['#00ff88', '#4dff88', '#ff6b6b', '#ffa500']),
        )])
        fig_pie.update_layout(paper_bgcolor='#0a0a0a', font_color='#c9d1d9', height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown("### Winner/Error by Shot")
        p1 = data['p1_stats']
        shot_df = pd.DataFrame({
            'Shot': ['FH', 'BH', 'Serve', 'Volley'] * 2,
            'Type': ['Winners'] * 4 + ['Errors'] * 4,
            'Count': [
                p1.get('forehand_winners', 0), p1.get('backhand_winners', 0),
                p1.get('aces', 0), p1.get('net_points_won', 0),
                p1.get('forehand_errors', 0), p1.get('backhand_errors', 0),
                p1.get('double_faults', 0), max(0, p1.get('net_points_total', 0) - p1.get('net_points_won', 0)),
            ],
        })
        fig_bar = px.bar(shot_df, x='Shot', y='Count', color='Type', barmode='group',
                         color_discrete_map={'Winners': '#00ff88', 'Errors': '#ff6b6b'})
        fig_bar.update_layout(paper_bgcolor='#0a0a0a', plot_bgcolor='#0d1117', font_color='#c9d1d9', height=350)
        st.plotly_chart(fig_bar, use_container_width=True)

# ── Match Review ─────────────────────────────────────────────────────────────
elif "Match Review" in view:
    st.markdown("## Match Review")

    review = data['review']

    # Match observations
    st.markdown('<div class="section-header">Match Summary</div>', unsafe_allow_html=True)
    for obs in review.get('observations', []):
        st.markdown(f"- {obs}")

    st.markdown("---")

    # Per-player corrections
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {data['match']['player1']}")
        st.markdown('<div class="section-header">Corrections</div>', unsafe_allow_html=True)
        for i, correction in enumerate(review.get('p1_corrections', []), 1):
            st.markdown(f"**{i}.** {correction}")

    with col2:
        st.markdown(f"### {data['match']['player2']}")
        st.markdown('<div class="section-header">Corrections</div>', unsafe_allow_html=True)
        for i, correction in enumerate(review.get('p2_corrections', []), 1):
            st.markdown(f"**{i}.** {correction}")

    st.markdown("---")

    # Line call history
    st.markdown('<div class="section-header">Line Call History</div>', unsafe_allow_html=True)
    lc_data = data['line_calls']['calls']
    if lc_data:
        lc_df = pd.DataFrame(lc_data)
        st.dataframe(lc_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No line calls recorded")

    st.markdown("---")

    # Suggested drills
    st.markdown('<div class="section-header">Recommended Practice</div>', unsafe_allow_html=True)
    for i, drill in enumerate(review.get('drills', []), 1):
        with st.expander(f"Drill {i}: {drill.split('(')[0].strip()}"):
            st.write(drill)

# ── Trends ───────────────────────────────────────────────────────────────────
elif "Trends" in view:
    st.markdown("## Performance Trends")

    dates = pd.date_range(end=datetime.now(), periods=12, freq='W')
    trend_df = pd.DataFrame({
        'Date': dates,
        'First Serve %': np.random.normal(65, 5, 12).clip(40, 85),
        'Winner/UE Ratio': np.random.normal(1.2, 0.3, 12).clip(0.3, 2.5),
        'Win Rate %': np.cumsum(np.random.choice([-2, 3, 5], 12)).clip(40, 90),
    })

    fig_trend = px.line(trend_df, x='Date', y=['First Serve %', 'Win Rate %'],
                        color_discrete_sequence=['#00ff88', '#ff6b6b'])
    fig_trend.update_layout(paper_bgcolor='#0a0a0a', plot_bgcolor='#0d1117', font_color='#c9d1d9', height=400)
    st.plotly_chart(fig_trend, use_container_width=True)

    # Summary stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches Played", "12")
    c2.metric("Win Rate", "67%", "+8%")
    c3.metric("Avg Rating", "6.8", "+0.3")
    c4.metric("Consistency", "72%", "+4%")

    # Line call accuracy over time
    st.markdown('<div class="section-header">Line Call Confidence Trend</div>', unsafe_allow_html=True)
    lc_trend = pd.DataFrame({
        'Match': [f"Match {i}" for i in range(1, 13)],
        'Avg Confidence %': np.random.normal(88, 4, 12).clip(75, 99),
        'Total Calls': np.random.randint(10, 25, 12),
    })
    fig_lc = px.bar(lc_trend, x='Match', y='Total Calls', color='Avg Confidence %',
                    color_continuous_scale='Viridis')
    fig_lc.update_layout(paper_bgcolor='#0a0a0a', plot_bgcolor='#0d1117', font_color='#c9d1d9', height=300)
    st.plotly_chart(fig_lc, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("TennisIQ v1.0.0")
