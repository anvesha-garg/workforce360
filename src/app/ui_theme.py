"""Workforce360 glassmorphism theme.

Drop this file next to your dashboard script (or in src/ui/) and call
``inject_theme()`` once, right after ``st.set_page_config(...)``.
All existing class names (w360-hero, w360-pill, w360-feature, w360-card)
are preserved, so no markup or data logic needs to change.
"""

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Shared palette -- reuse in charts if you like.
W360_SCALE = ["#134E4A", "#14B8A6", "#A5F3FC"]   # low -> high, readable on dark
W360_WARM_SCALE = ["#78350F", "#F59E0B", "#FDE68A"]
W360_COLORWAY = ["#22D3EE", "#818CF8", "#2DD4BF", "#F59E0B", "#F43F5E", "#A78BFA"]

# Plotly template: transparent canvas, light text, soft grid. Applied globally,
# so every existing px chart picks it up with no code changes.
pio.templates["w360"] = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, system-ui, sans-serif", color="#E2E8F0", size=13),
        colorway=W360_COLORWAY,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(size=16, color="#F8FAFC")),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.12)",
                   linecolor="rgba(255,255,255,0.15)", title_font=dict(color="#94A3B8")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.12)",
                   linecolor="rgba(255,255,255,0.15)", title_font=dict(color="#94A3B8")),
        legend=dict(bgcolor="rgba(255,255,255,0.06)", bordercolor="rgba(255,255,255,0.14)", borderwidth=1),
        hoverlabel=dict(bgcolor="#0B1B33", bordercolor="#22D3EE", font=dict(color="#F8FAFC")),
        coloraxis=dict(colorbar=dict(outlinewidth=0, tickfont=dict(color="#94A3B8"))),
    )
)
pio.templates.default = "w360"


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #060E1C;
    --text: #F1F5F9;
    --muted: rgba(203, 213, 225, 0.72);
    --accent: #22D3EE;
    --accent-2: #818CF8;
    --glass: linear-gradient(135deg, rgba(255,255,255,0.13), rgba(255,255,255,0.045));
    --glass-flat: rgba(255, 255, 255, 0.07);
    --border: rgba(255, 255, 255, 0.16);
    --border-glow: rgba(165, 243, 252, 0.34);
    --shadow: 0 14px 40px rgba(0, 0, 0, 0.30);
    --blur: blur(18px) saturate(150%);
    --radius: 20px;
}

html, body, [class*="css"], .stApp, button, input, textarea, select {
    font-family: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
}

/* ---------- Canvas ---------- */
.stApp {
    background:
        radial-gradient(circle at 8% 6%,  rgba(34, 211, 238, 0.26), transparent 30%),
        radial-gradient(circle at 94% 10%, rgba(99, 102, 241, 0.28), transparent 32%),
        radial-gradient(circle at 50% 105%, rgba(20, 184, 166, 0.22), transparent 36%),
        radial-gradient(circle at 80% 60%, rgba(167, 139, 250, 0.10), transparent 30%),
        var(--bg);
    background-attachment: fixed;
    color: var(--text);
}
.block-container {
    max-width: 1380px;
    padding: 2rem 2rem 7rem;
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] {
    background: rgba(6, 14, 28, 0.55);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
}

/* ---------- Typography ---------- */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 { letter-spacing: -0.02em; }
.stApp h1 { font-size: 2.3rem; font-weight: 800; color: #fff; }
.stApp h2, .stApp [data-testid="stHeading"] h2 { font-size: 1.4rem; font-weight: 700; color: #A5F3FC; }
.stApp h3 { font-size: 1.1rem; font-weight: 700; color: #E2E8F0; }
.stApp h4 { font-size: 1rem; font-weight: 600; color: #CBD5E1; }
.stApp p, .stApp li, .stApp label, [data-testid="stMarkdownContainer"] { color: var(--text); }
[data-testid="stCaptionContainer"], .stApp small { color: var(--muted) !important; }
hr { border-color: var(--border) !important; margin: 1.4rem 0 !important; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: rgba(6, 14, 28, 0.70);
    border-right: 1px solid var(--border);
    backdrop-filter: var(--blur);
    -webkit-backdrop-filter: var(--blur);
}
section[data-testid="stSidebar"] h2 { color: #fff !important; font-size: 1.25rem !important; }
section[data-testid="stSidebar"] h3 {
    font-size: 0.74rem !important; text-transform: uppercase;
    letter-spacing: 0.12em; color: var(--muted) !important;
}

/* ---------- Hero ---------- */
.w360-hero {
    position: relative; overflow: hidden;
    padding: 2.6rem 2.8rem; margin-bottom: 1.8rem;
    border-radius: 28px;
    background:
        linear-gradient(135deg, rgba(34,211,238,0.24), rgba(99,102,241,0.20)),
        rgba(255,255,255,0.06);
    border: 1px solid var(--border-glow);
    box-shadow: 0 28px 70px rgba(0,0,0,0.34), inset 0 1px 0 rgba(255,255,255,0.20);
    backdrop-filter: blur(24px) saturate(160%);
    -webkit-backdrop-filter: blur(24px) saturate(160%);
}
.w360-hero::before, .w360-hero::after {
    content: ""; position: absolute; border-radius: 50%; filter: blur(14px); pointer-events: none;
}
.w360-hero::before { width: 300px; height: 300px; right: -90px; top: -120px; background: rgba(103,232,249,0.22); }
.w360-hero::after  { width: 240px; height: 240px; left: 38%; bottom: -160px; background: rgba(129,140,248,0.18); }
.w360-hero > * { position: relative; z-index: 1; }
.w360-hero h1 {
    margin: 0 0 0.5rem; font-size: 2.7rem;
    background: linear-gradient(90deg, #fff 20%, #A5F3FC 70%, #C7D2FE);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.w360-hero p { margin: 0; max-width: 720px; font-size: 1.05rem; line-height: 1.6; color: rgba(241,245,249,0.90) !important; }
.w360-pill {
    display: inline-block; margin: 1.1rem 0.45rem 0 0; padding: 0.34rem 0.9rem;
    border-radius: 999px; font-size: 0.76rem; font-weight: 600; letter-spacing: 0.02em;
    color: #E0F2FE !important;
    background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.26);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
}

/* ---------- Cards ---------- */
.w360-feature, .w360-card {
    height: 100%; padding: 1.25rem 1.3rem;
    background: var(--glass); border: 1px solid var(--border); border-radius: var(--radius);
    box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,0.10);
    backdrop-filter: var(--blur); -webkit-backdrop-filter: var(--blur);
    transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}
.w360-feature { border-top: 3px solid var(--accent); }
.w360-feature:hover, .w360-card:hover {
    transform: translateY(-4px); border-color: var(--border-glow);
    box-shadow: 0 22px 50px rgba(0,0,0,0.38), 0 0 0 1px rgba(34,211,238,0.10);
}
.w360-feature h5 { margin: 0 0 0.5rem; font-size: 0.98rem; font-weight: 700; color: #A5F3FC !important; }
.w360-feature p  { margin: 0; font-size: 0.86rem; line-height: 1.55; color: var(--muted) !important; }

/* ---------- Metrics ---------- */
div[data-testid="stMetric"] {
    padding: 1rem 1.15rem; border-radius: 18px;
    background: var(--glass); border: 1px solid var(--border);
    box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,0.10);
    backdrop-filter: var(--blur); -webkit-backdrop-filter: var(--blur);
    transition: transform .2s ease, border-color .2s ease;
}
div[data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: var(--border-glow); }
div[data-testid="stMetricLabel"] p {
    font-size: 0.74rem !important; font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted) !important;
}
div[data-testid="stMetricValue"] { font-size: 1.75rem !important; font-weight: 800 !important; color: #fff !important; }
div[data-testid="stMetricDelta"] { font-weight: 600; }

/* ---------- Tabs (scrollable, so 8 tabs never wrap) ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; padding: 6px; border-radius: 18px;
    background: var(--glass-flat); border: 1px solid var(--border);
    backdrop-filter: var(--blur); -webkit-backdrop-filter: var(--blur);
    overflow-x: auto; scrollbar-width: none;
}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab"] {
    height: auto; padding: 0.6rem 1.05rem; border-radius: 13px; white-space: nowrap;
    font-weight: 600; color: var(--muted) !important; transition: all .2s ease;
}
.stTabs [data-baseweb="tab"]:hover { background: rgba(255,255,255,0.08); color: #fff !important; }
.stTabs [data-baseweb="tab"] p { color: inherit !important; font-size: 0.92rem; }
.stTabs [aria-selected="true"] {
    color: #CFFAFE !important;
    background: linear-gradient(135deg, rgba(34,211,238,0.24), rgba(99,102,241,0.20)) !important;
    border: 1px solid rgba(165,243,252,0.32);
    box-shadow: 0 8px 22px rgba(34,211,238,0.14), inset 0 1px 0 rgba(255,255,255,0.16);
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.4rem; }
/* nested tabs (compensation) -- slightly quieter */
.stTabs .stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.045); }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
    padding: 0.55rem 1.3rem; border-radius: 14px; font-weight: 700; color: #fff !important;
    background: linear-gradient(135deg, rgba(34,211,238,0.80), rgba(99,102,241,0.80));
    border: 1px solid rgba(255,255,255,0.26);
    box-shadow: 0 10px 26px rgba(34,211,238,0.16);
    transition: transform .2s ease, box-shadow .2s ease, filter .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px); filter: brightness(1.1);
    box-shadow: 0 16px 34px rgba(34,211,238,0.28); border-color: rgba(255,255,255,0.4);
}
.stButton > button:active { transform: translateY(0); }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
    outline: 2px solid #A5F3FC; outline-offset: 2px;
}

/* ---------- Inputs ---------- */
input, textarea, div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="base-input"] {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid var(--border) !important;
    border-radius: 13px !important; color: var(--text) !important;
    transition: border-color .2s ease, box-shadow .2s ease;
}
input:focus, textarea:focus, div[data-baseweb="select"] > div:focus-within {
    border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(34,211,238,0.18) !important;
}
input::placeholder, textarea::placeholder { color: rgba(226,232,240,0.5) !important; }
[data-testid="stWidgetLabel"] p { font-size: 0.84rem; font-weight: 600; color: #CBD5E1 !important; }
div[data-baseweb="popover"] ul { background: #0B1B33 !important; border: 1px solid var(--border); border-radius: 12px; }

/* sliders */
div[data-baseweb="slider"] [role="slider"] {
    background: linear-gradient(135deg, #22D3EE, #818CF8) !important;
    border: 2px solid #fff !important; box-shadow: 0 0 0 5px rgba(34,211,238,0.18);
}
div[data-testid="stSlider"] [data-testid="stThumbValue"] { color: #A5F3FC !important; font-weight: 700; }

/* progress */
div[data-testid="stProgress"] > div > div { background: rgba(255,255,255,0.10) !important; border-radius: 999px; }
div[data-testid="stProgress"] > div > div > div { background: linear-gradient(90deg, #22D3EE, #818CF8) !important; border-radius: 999px; }

/* forms + expanders as glass panels */
div[data-testid="stForm"], div[data-testid="stExpander"] details {
    background: var(--glass-flat); border: 1px solid var(--border); border-radius: var(--radius);
    backdrop-filter: var(--blur); -webkit-backdrop-filter: var(--blur);
}
div[data-testid="stForm"] { padding: 1.3rem; }
div[data-testid="stExpander"] summary { font-weight: 600; padding: 0.7rem 1rem; }
div[data-testid="stExpander"] summary:hover { color: #A5F3FC; }

/* ---------- Data + charts ---------- */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border); border-radius: 18px; overflow: hidden;
    background: var(--glass-flat); box-shadow: var(--shadow);
}
div[data-testid="stPlotlyChart"], div[data-testid="stPyplot"] {
    padding: 0.6rem; border-radius: var(--radius);
    background: var(--glass-flat); border: 1px solid var(--border);
    box-shadow: var(--shadow); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
}
div[data-testid="stPyplot"] img { border-radius: 14px; background: #fff; padding: 6px; }

/* ---------- Alerts: tinted glass ---------- */
div[data-testid="stAlert"] {
    border-radius: 16px; border: 1px solid var(--border);
    backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
}
div[data-testid="stAlert"][data-baseweb="notification"] { background: rgba(255,255,255,0.08); }
div[data-testid="stAlertContentError"]   { color: #FECDD3; }
div[data-testid="stAlertContentSuccess"] { color: #A7F3D0; }
div[data-testid="stAlertContentWarning"] { color: #FDE68A; }
div[data-testid="stAlertContentInfo"]    { color: #BAE6FD; }

/* ---------- Dialog + chat ---------- */
div[data-testid="stDialog"] div[role="dialog"] {
    background: rgba(8, 18, 36, 0.88) !important;
    border: 1px solid var(--border-glow); border-radius: 26px !important;
    box-shadow: 0 30px 90px rgba(0,0,0,0.5);
    backdrop-filter: blur(26px) saturate(160%); -webkit-backdrop-filter: blur(26px) saturate(160%);
}
div[data-testid="stChatMessage"] {
    background: var(--glass-flat); border: 1px solid var(--border);
    border-radius: 18px; padding: 0.55rem 0.9rem; margin-bottom: 0.7rem;
}
div[data-testid="stChatInput"] { border-radius: 16px; }

/* ---------- Floating copilot launcher ---------- */
div[data-testid="stButton"] button[kind="primary"] {
    position: fixed !important; right: 28px !important; bottom: 28px !important;
    width: 64px !important; height: 64px !important; min-width: 64px !important;
    padding: 0 !important; border-radius: 50% !important; font-size: 26px !important;
    z-index: 999999 !important;
    background: linear-gradient(135deg, rgba(34,211,238,0.92), rgba(99,102,241,0.92)) !important;
    border: 1px solid rgba(255,255,255,0.40) !important;
    box-shadow: 0 0 0 8px rgba(103,232,249,0.10), 0 20px 48px rgba(0,0,0,0.40),
                inset 0 1px 0 rgba(255,255,255,0.32) !important;
    animation: w360-pulse 3s ease-in-out infinite;
}
div[data-testid="stButton"] button[kind="primary"]:hover { transform: scale(1.1) !important; animation: none; }
@keyframes w360-pulse {
    0%, 100% { box-shadow: 0 0 0 6px rgba(103,232,249,0.10), 0 20px 48px rgba(0,0,0,0.40); }
    50%      { box-shadow: 0 0 0 14px rgba(103,232,249,0.04), 0 20px 48px rgba(0,0,0,0.40); }
}

/* ---------- Scrollbars ---------- */
* { scrollbar-width: thin; scrollbar-color: rgba(165,243,252,0.30) transparent; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: rgba(165,243,252,0.28); border-radius: 8px; }

/* ---------- Responsive ---------- */
@media (max-width: 700px) {
    .block-container { padding: 1rem 0.75rem 6rem; }
    .w360-hero { padding: 1.6rem; border-radius: 22px; }
    .w360-hero h1 { font-size: 2rem; }
    div[data-testid="stButton"] button[kind="primary"] { right: 18px !important; bottom: 18px !important; }
}
@media (prefers-reduced-motion: reduce) {
    * { animation: none !important; transition: none !important; }
}
</style>
"""


def inject_theme() -> None:
    """Inject the Workforce360 glassmorphism stylesheet."""
    st.markdown(_CSS, unsafe_allow_html=True)


# =====================================================================
# Layout switcher: tab placement + panel arrangement
# =====================================================================
NAV_MODES = ["Top tabs", "Left menu", "Bottom bar", "Sidebar menu"]
PANEL_MODES = ["Metrics on top", "Metrics left", "Charts first"]

_LAYOUT_BASE_CSS = """
[class*="st-key-w360_hide_"] { display: none !important; }
"""

_NAV_CSS = {
    "Top tabs": "",
    "Left menu": """
.st-key-w360_tabs [data-baseweb="tabs"] { display: flex; flex-direction: row; align-items: flex-start; gap: 1.6rem; }
.st-key-w360_tabs [data-baseweb="tabs"] > [data-baseweb="tab-list"] {
    flex: 0 0 235px; flex-direction: column; align-items: stretch; overflow: visible;
    position: sticky; top: 4.5rem; padding: 10px; gap: 6px;
}
.st-key-w360_tabs [data-baseweb="tabs"] > [data-baseweb="tab-panel"] { flex: 1 1 0; min-width: 0; padding-top: 0; }
.st-key-w360_tabs [data-baseweb="tabs"] > [data-baseweb="tab-list"] [data-baseweb="tab"] {
    justify-content: flex-start; width: 100%; padding: 0.75rem 1rem;
}
/* nested tabs (e.g. Compensation) stay horizontal */
.st-key-w360_tabs [data-baseweb="tab-panel"] [data-baseweb="tabs"] { display: block; }
.st-key-w360_tabs [data-baseweb="tab-panel"] [data-baseweb="tab-list"] {
    position: static; flex-direction: row; padding: 6px; overflow-x: auto;
}
.st-key-w360_tabs [data-baseweb="tab-panel"] [data-baseweb="tab"] { width: auto; }
@media (max-width: 900px) {
    .st-key-w360_tabs [data-baseweb="tabs"] { flex-direction: column; }
    .st-key-w360_tabs [data-baseweb="tabs"] > [data-baseweb="tab-list"] { flex-direction: row; position: static; overflow-x: auto; }
}
""",
    "Bottom bar": """
.st-key-w360_tabs [data-baseweb="tabs"] > [data-baseweb="tab-list"] {
    position: fixed; left: 50%; transform: translateX(-50%); bottom: 16px; z-index: 99990;
    max-width: calc(100vw - 32px); justify-content: flex-start; padding: 8px;
    background: rgba(8, 18, 36, 0.78); box-shadow: 0 18px 50px rgba(0,0,0,0.5);
}
.st-key-w360_tabs [data-baseweb="tab-panel"] [data-baseweb="tab-list"] {
    position: static; transform: none; max-width: none; box-shadow: none;
    background: rgba(255,255,255,0.045); z-index: auto;
}
.block-container { padding-bottom: 9rem !important; }
div[data-testid="stButton"] button[kind="primary"] { bottom: 96px !important; }
""",
    "Sidebar menu": """
.st-key-w360_page [role="radiogroup"] { gap: 6px; }
.st-key-w360_page label {
    width: 100%; margin: 0; padding: 0.7rem 0.95rem; border-radius: 13px; cursor: pointer;
    background: rgba(255,255,255,0.06); border: 1px solid var(--border); transition: all .2s ease;
}
.st-key-w360_page label > div:first-child { display: none; }
.st-key-w360_page label:hover { background: rgba(255,255,255,0.12); transform: translateX(3px); }
.st-key-w360_page label:has(input:checked) {
    background: linear-gradient(135deg, rgba(34,211,238,0.26), rgba(99,102,241,0.22));
    border-color: rgba(165,243,252,0.40); box-shadow: 0 8px 22px rgba(34,211,238,0.14);
}
.st-key-w360_page label p { font-weight: 600; font-size: 0.92rem; }
""",
}


def layout_controls():
    """Render the layout pickers in the sidebar and return (nav_mode, panel_mode)."""
    st.markdown("### Layout")
    nav_mode = st.selectbox("Tab placement", NAV_MODES, key="w360_nav_mode")
    panel_mode = st.selectbox("Panel arrangement", PANEL_MODES, key="w360_panel_mode")
    return nav_mode, panel_mode


def inject_layout_css(nav_mode: str) -> None:
    css = _LAYOUT_BASE_CSS + _NAV_CSS.get(nav_mode, "")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def layout_tabs(labels, nav_mode: str):
    """Drop-in replacement for st.tabs(labels) honouring the chosen placement.

    Returns one context manager per label, so `with overview_tab:` keeps working.
    """
    if nav_mode == "Sidebar menu":
        with st.sidebar:
            st.markdown("### Navigate")
            choice = st.radio("Page", labels, key="w360_page", label_visibility="collapsed")
        return [
            st.container() if label == choice else st.container(key=f"w360_hide_{index}")
            for index, label in enumerate(labels)
        ]
    with st.container(key="w360_tabs"):
        return st.tabs(labels)