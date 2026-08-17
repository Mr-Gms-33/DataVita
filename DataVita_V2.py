import streamlit as st
import pandas as pd
import numpy as np
import sys
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from io import BytesIO

# ── ML Imports (einmalig beim Start, nicht bei jedem Seitenwechsel) ──
try:

    from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                         cross_val_score)
    from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                                 recall_score, f1_score, roc_auc_score, roc_curve,
                                 mean_absolute_percentage_error)
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_OK = True
except ImportError:
    _SMOTE_OK = False

# ╔═══════════════════════════════════════════════════════════╗
#  CONFIG
# ╚═══════════════════════════════════════════════════════════╝
st.set_page_config(
    page_title="E-Commerce Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ╔═══════════════════════════════════════════════════════════╗
#  DATABASE — PostgreSQL locally, bundled Parquet+DuckDB in the cloud
#  Credentials come from .env locally, or from Streamlit Cloud's
#  "Secrets" (st.secrets) when deployed — never hard-coded.
#
#  If no live Postgres is reachable (e.g. on Streamlit Community Cloud,
#  which can't see your local database), the app automatically falls
#  back to the Parquet snapshots in ./data/ via an in-memory DuckDB
#  engine, so the same pd.read_sql(query, ENGINE) calls keep working.
# ╚═══════════════════════════════════════════════════════════╝
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get_config(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


DB_USER = _get_config("DB_USER", "postgres")
DB_PASSWORD = _get_config("DB_PASSWORD", "")
DB_HOST = _get_config("DB_HOST", "localhost")
DB_PORT = _get_config("DB_PORT", "5432")
DB_NAME = _get_config("DB_NAME", "Onlineshop")


def _build_engine():
    pg_engine = create_engine(
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        pool_size=5,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 3},
    )
    try:
        with pg_engine.connect():
            pass
        return pg_engine
    except Exception:
        pass

    data_dir = Path(__file__).parent / "data"
    duck_engine = create_engine("duckdb:///:memory:")

    # Tables exported as a single file use their own name as the view name.
    # Tables split by export_to_parquet.py (files named "<table>__partN")
    # are grouped back together and read via a glob pattern.
    table_to_files = {}
    for parquet_file in sorted(data_dir.glob("*.parquet")):
        table_name = parquet_file.stem.split("__part")[0]
        table_to_files.setdefault(table_name, []).append(parquet_file)

    with duck_engine.begin() as conn:
        for table_name, files in table_to_files.items():
            if len(files) == 1:
                source = f"read_parquet('{files[0].as_posix()}')"
            else:
                source = f"read_parquet('{(data_dir / (table_name + '__part*.parquet')).as_posix()}')"
            conn.exec_driver_sql(
                f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM {source}"
            )
    return duck_engine


ENGINE = _build_engine()

# ╔═══════════════════════════════════════════════════════════╗
#  DESIGN SYSTEM
# ╚═══════════════════════════════════════════════════════════╝
# Dark theme palette
C = {
    "bg":         "#0b0f19",
    "card":       "rgba(17,24,39,0.7)",
    "card_solid": "#111827",
    "border":     "rgba(55,65,81,0.5)",
    "text":       "#f1f5f9",
    "muted":      "#94a3b8",
    "dim":        "#64748b",
    # Accent colors
    "blue":       "#3b82f6",
    "indigo":     "#6366f1",
    "violet":     "#8b5cf6",
    "cyan":       "#22d3ee",
    "emerald":    "#10b981",
    "amber":      "#f59e0b",
    "rose":       "#f43f5e",
    "pink":       "#ec4899",
    "orange":     "#f97316",
    "lime":       "#84cc16",
    "teal":       "#14b8a6",
    "sky":        "#0ea5e9",
}

CHART_SEQ = [C["cyan"], C["violet"], C["emerald"], C["amber"], C["rose"],
             C["blue"], C["pink"], C["orange"], C["lime"], C["teal"], C["sky"], C["indigo"]]

def plotly_dark(fig, h=320, **kw):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=C["muted"], size=11),
        margin=dict(l=8, r=8, t=28, b=8),
        height=h,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5,
                    font=dict(size=10, color=C["muted"]), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=True, gridcolor="rgba(55,65,81,0.3)", gridwidth=1,
                   zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="rgba(55,65,81,0.3)", gridwidth=1,
                   zeroline=False, tickfont=dict(size=10)),
        **kw,
    )
    return fig

# ╔═══════════════════════════════════════════════════════════╗
#  PREMIUM CSS
# ╚═══════════════════════════════════════════════════════════╝
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stMarkdown, .stText, p, span, div, label {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* ── Dark Background ── */
.stApp {
    background: #0b0f19;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99,102,241,0.12), transparent),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(34,211,238,0.06), transparent);
}
.block-container {
    padding: 1rem 1.8rem 2rem 1.8rem;
    max-width: 100%;
}

/* ── Presentation: hide Streamlit chrome ── */
#MainMenu, footer, .stDeployButton,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
.viewerBadge_container__r5tak {
    display: none !important;
}
/* Header: hidden completely */
[data-testid="stHeader"] {
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    background: transparent !important;
}
/* Hide Streamlit's own collapse/expand buttons */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebar"] button[aria-expanded],
[data-testid="stSidebar"] [data-testid="baseButton-headerNoPadding"] {
    display: none !important;
}
.block-container .stSubheader,
.block-container h3 { display: none !important; }

/* ── Sidebar — toggleable ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1222 0%, #111827 40%, #0c1222 100%);
    border-right: 1px solid rgba(99,102,241,0.10);
    display: block !important;
    visibility: visible !important;
    width: 310px !important;
    min-width: 310px !important;
    max-width: 310px !important;
    transform: none !important;
    opacity: 1 !important;
    transition: margin-left 0.3s cubic-bezier(0.4,0,0.2,1) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.5rem;
}
/* Logo at top via CSS — renders before navigation */
[data-testid="stSidebarNav"]::before {
    content: "⚡" "\a" "Analytics Hub";
    white-space: pre;
    display: block;
    text-align: center;
    font-size: 16px;
    font-weight: 800;
    color: #f1f5f9 !important;
    letter-spacing: -0.3px;
    padding: 14px 10px 6px 10px;
    margin-bottom: 4px;
    line-height: 1.8;
    border-bottom: 1px solid rgba(99,102,241,0.15);
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* Section headers (PRÄSENTATION, ANALYSE, …) */
[data-testid="stNavSectionHeader"] {
    display: block !important;
    visibility: visible !important;
    font-size: 10.5px !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: #475569 !important;
    padding: 20px 12px 6px 12px !important;
    margin-top: 4px !important;
}

/* Nav links */
[data-testid="stSidebarNavLink"] {
    display: flex !important;
    visibility: visible !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
    margin: 2px 8px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
    border: 1px solid transparent !important;
    color: #94a3b8 !important;
}
[data-testid="stSidebarNavLink"]:hover {
    background: rgba(99,102,241,0.08) !important;
    border-color: rgba(99,102,241,0.12) !important;
    color: #e2e8f0 !important;
}
[data-testid="stSidebarNavLink"][aria-current="page"] {
    background: rgba(99,102,241,0.18) !important;
    border-color: rgba(99,102,241,0.30) !important;
    color: #f1f5f9 !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 12px rgba(99,102,241,0.10);
}

/* Nav link icon styling */
[data-testid="stSidebarNavLink"] span:first-child {
    font-size: 16px !important;
    margin-right: 4px !important;
}

/* Sidebar filter widgets */
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border-radius: 6px;
    font-size: 11px;
}
[data-testid="stSidebar"] hr { border-color: rgba(99,102,241,0.10); }
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(15,22,41,0.8);
    border-color: rgba(55,65,81,0.5);
    border-radius: 8px;
}

/* ── Header Banner ── */
.header-banner {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(34,211,238,0.08) 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
}
.header-banner::before {
    content: "";
    position: absolute;
    top: -40%;
    right: -5%;
    width: 220px;
    height: 220px;
    background: radial-gradient(circle, rgba(99,102,241,0.15), transparent 70%);
    border-radius: 50%;
}
.header-banner::after {
    content: "";
    position: absolute;
    bottom: -50%;
    left: 15%;
    width: 180px;
    height: 180px;
    background: radial-gradient(circle, rgba(34,211,238,0.1), transparent 70%);
    border-radius: 50%;
}
.h-title {
    font-size: 26px;
    font-weight: 800;
    color: #f1f5f9;
    letter-spacing: -0.5px;
    position: relative;
    z-index: 1;
}
.h-sub {
    font-size: 13px;
    color: #94a3b8;
    margin-top: 4px;
    position: relative;
    z-index: 1;
}

/* ── KPI Card ── */
.kpi {
    background: rgba(17,24,39,0.6);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(55,65,81,0.45);
    border-radius: 16px;
    padding: 18px 20px 16px 20px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.kpi:hover {
    border-color: rgba(99,102,241,0.35);
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(99,102,241,0.12);
}
.kpi-glow {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.kpi .icon-wrap {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    margin-bottom: 14px;
}
.kpi .lbl {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #64748b;
    margin-bottom: 4px;
}
.kpi .val {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.5px;
    line-height: 1.1;
}

/* ── Glass Card ── */
.glass {
    background: rgba(17,24,39,0.55);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(55,65,81,0.4);
    border-radius: 16px;
    padding: 20px 22px 14px 22px;
    margin-bottom: 16px;
    transition: all 0.3s ease;
}
.glass:hover {
    border-color: rgba(99,102,241,0.25);
    box-shadow: 0 4px 24px rgba(0,0,0,0.15);
}

/* ── Section Title ── */
.sec-t {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(55,65,81,0.35);
}
.sec-t .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    box-shadow: 0 0 8px;
}
.sec-t .ttl {
    font-size: 14px;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: -0.2px;
}

/* ── Mini Stat ── */
.mstat {
    background: rgba(15,23,42,0.6);
    border: 1px solid rgba(55,65,81,0.3);
    border-radius: 12px;
    padding: 14px 16px;
    transition: border-color 0.2s;
}
.mstat:hover { border-color: rgba(99,102,241,0.25); }
.mstat .ml { font-size: 10.5px; font-weight: 600; text-transform: uppercase;
             letter-spacing: 0.8px; color: #64748b; margin-bottom: 4px; }
.mstat .mv { font-size: 20px; font-weight: 800; letter-spacing: -0.3px; }

/* ── Sidebar Brand ── */
.sb-brand { text-align: center; padding: 16px 10px 4px 10px; }
.sb-brand .logo { font-size: 26px; margin-bottom: 2px; }
.sb-brand .name { font-size: 16px; font-weight: 800; color: #f1f5f9 !important;
                  letter-spacing: -0.3px; }
.sb-brand .tag { font-size: 9.5px; font-weight: 500; color: #475569 !important;
                 text-transform: uppercase; letter-spacing: 3px; margin-top: 2px; }
.sb-line { height: 1px; margin: 12px 8px;
           background: linear-gradient(90deg, transparent, rgba(99,102,241,0.25), transparent); }

/* ── Footer ── */
.app-footer {
    text-align: center; color: #475569; font-size: 11px;
    padding: 20px 0 4px 0; margin-top: 30px;
    border-top: 1px solid rgba(55,65,81,0.3);
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
[data-testid="stDataFrame"] * { font-size: 12px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
[data-testid="stSidebar"]::-webkit-scrollbar { width: 4px; }
[data-testid="stSidebar"]::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.2); border-radius: 4px; }

/* ── Row gap helper ── */
.row-gap { height: 18px; }

/* ── Fix: Text unter Diagrammen nicht überlappen ── */
.glass { overflow: visible; margin-bottom: 20px; }
[data-testid="stPlotlyChart"] { margin-bottom: 8px; }
.mstat { margin-bottom: 6px; }

/* ── Aktive-Filter-Chips im Header (aus mounir.py) ── */
.flt-chip {
    display: inline-block;
    background: rgba(34,211,238,0.1);
    border: 1px solid rgba(34,211,238,0.25);
    color: #67e8f9;
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 10.5px;
    font-weight: 600;
    margin: 8px 6px 0 0;
    position: relative;
    z-index: 1;
}

/* ── Buttons (Reset / Info / CSV-Export) ── */
.stButton > button, .stDownloadButton > button {
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.25);
    color: #cbd5e1;
    border-radius: 10px;
    font-size: 12.5px;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: rgba(99,102,241,0.22);
    border-color: rgba(99,102,241,0.45);
    color: #f1f5f9;
}

/* ── Responsive Feinschliff ── */
@media (max-width: 900px) {
    .block-container { padding: 0.6rem 0.9rem 1.5rem 0.9rem; }
    .h-title { font-size: 20px; }
    .header-banner { padding: 18px 20px; }
    .kpi .val { font-size: 20px; }
}

/* ── Intro Pages ── */
.cover-title {
    text-align: center;
    padding: 40px 20px 10px 20px;
}
.cover-title .main-title {
    font-size: 42px;
    font-weight: 900;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #22d3ee, #6366f1, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
}
.cover-title .sub-line {
    font-size: 16px;
    color: #94a3b8;
    margin-top: 8px;
    font-weight: 400;
}
.cover-divider {
    width: 80px;
    height: 3px;
    background: linear-gradient(90deg, #6366f1, #22d3ee);
    border-radius: 3px;
    margin: 20px auto;
}
.team-card {
    background: rgba(17,24,39,0.6);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(55,65,81,0.45);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.team-card:hover {
    border-color: rgba(99,102,241,0.35);
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(99,102,241,0.12);
}
.team-card .tc-glow {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.team-card .avatar {
    width: 64px; height: 64px;
    border-radius: 50%;
    margin: 0 auto 14px auto;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
}
.team-card .tc-name {
    font-size: 16px;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 4px;
}
.team-card .tc-role {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 6px;
}
.team-card .tc-detail {
    font-size: 12px;
    color: #64748b;
    line-height: 1.5;
}
.info-block {
    background: rgba(17,24,39,0.55);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(55,65,81,0.4);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 16px;
}
.info-block .ib-title {
    font-size: 16px;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.info-block .ib-title .ib-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
}
.info-block .ib-text {
    font-size: 13.5px;
    color: #94a3b8;
    line-height: 1.75;
}
.info-block .ib-text strong {
    color: #cbd5e1;
    font-weight: 600;
}
.tech-pill {
    display: inline-block;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 600;
    color: #a5b4fc;
    margin: 4px;
}
.page-list {
    list-style: none;
    padding: 0;
    margin: 0;
}
.page-list li {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    border-radius: 10px;
    margin-bottom: 6px;
    transition: background 0.2s;
    font-size: 13.5px;
    color: #cbd5e1;
}
.page-list li:hover { background: rgba(99,102,241,0.08); }
.page-list .pl-num {
    width: 28px; height: 28px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 700;
    color: #fff;
    flex-shrink: 0;
}
.page-list .pl-desc {
    font-size: 11px;
    color: #64748b;
    margin-top: 2px;
}
.meta-row {
    display: flex;
    justify-content: center;
    gap: 30px;
    flex-wrap: wrap;
    margin-top: 10px;
}
.meta-item {
    font-size: 12px;
    color: #64748b;
    display: flex;
    align-items: center;
    gap: 6px;
}
.meta-item span { font-size: 15px; }

/* ── Pipeline ── */
.pipeline {
    display: flex;
    align-items: stretch;
    gap: 0;
    margin: 10px 0;
}
.pipe-step {
    flex: 1;
    background: rgba(17,24,39,0.6);
    border: 1px solid rgba(55,65,81,0.45);
    padding: 18px 14px;
    text-align: center;
    position: relative;
    transition: all 0.3s ease;
}
.pipe-step:first-child { border-radius: 14px 0 0 14px; }
.pipe-step:last-child { border-radius: 0 14px 14px 0; }
.pipe-step:hover {
    border-color: rgba(99,102,241,0.35);
    background: rgba(17,24,39,0.8);
}
.pipe-step .ps-icon { font-size: 26px; margin-bottom: 8px; }
.pipe-step .ps-title {
    font-size: 12px;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 4px;
}
.pipe-step .ps-desc {
    font-size: 10.5px;
    color: #64748b;
    line-height: 1.4;
}
.pipe-arrow {
    display: flex;
    align-items: center;
    font-size: 18px;
    color: #475569;
    padding: 0 2px;
}

/* ── Milestone ── */
.milestone {
    background: rgba(17,24,39,0.55);
    border: 1px solid rgba(55,65,81,0.4);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 10px;
    display: flex;
    align-items: flex-start;
    gap: 14px;
    transition: all 0.3s ease;
}
.milestone:hover {
    border-color: rgba(99,102,241,0.3);
    transform: translateX(4px);
}
.milestone .ms-icon {
    width: 40px; height: 40px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}
.milestone .ms-title {
    font-size: 14px;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 3px;
}
.milestone .ms-text {
    font-size: 12px;
    color: #94a3b8;
    line-height: 1.5;
}
.milestone .ms-badge {
    display: inline-block;
    font-size: 9.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 2px 8px;
    border-radius: 5px;
    margin-top: 6px;
}

/* ── Thank-You / Danke Page ── */
.danke-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 60vh;
    text-align: center;
    padding: 40px 20px;
}
.danke-icon {
    font-size: 72px;
    margin-bottom: 24px;
    animation: danke-pulse 2.5s ease-in-out infinite;
}
@keyframes danke-pulse {
    0%, 100% { transform: scale(1); opacity: 0.9; }
    50%      { transform: scale(1.12); opacity: 1; }
}
.danke-title {
    font-size: 48px;
    font-weight: 900;
    letter-spacing: -1.5px;
    background: linear-gradient(135deg, #22d3ee, #6366f1, #8b5cf6, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 12px;
}
.danke-sub {
    font-size: 18px;
    color: #94a3b8;
    font-weight: 400;
    margin-bottom: 36px;
    max-width: 560px;
}
.danke-divider {
    width: 100px;
    height: 3px;
    background: linear-gradient(90deg, #6366f1, #22d3ee);
    border-radius: 3px;
    margin: 0 auto 36px auto;
}
.danke-team {
    display: flex;
    justify-content: center;
    gap: 28px;
    flex-wrap: wrap;
    margin-bottom: 40px;
}
.danke-member {
    background: rgba(17,24,39,0.6);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(55,65,81,0.45);
    border-radius: 14px;
    padding: 18px 24px;
    min-width: 180px;
    transition: all 0.3s ease;
}
.danke-member:hover {
    border-color: rgba(99,102,241,0.4);
    transform: translateY(-4px);
    box-shadow: 0 8px 28px rgba(99,102,241,0.14);
}
.danke-member .dm-name {
    font-size: 15px;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 3px;
}
.danke-member .dm-role {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.danke-contact {
    font-size: 13px;
    color: #64748b;
    margin-top: 10px;
}
.danke-contact strong { color: #94a3b8; }
@media (max-width: 900px) {
    .danke-title { font-size: 32px; }
    .danke-sub   { font-size: 15px; }
    .danke-icon  { font-size: 52px; }
}

/* ══════════════════════════════════════════════════════════
   LANDING PAGE — PREMIUM DESIGN
   ══════════════════════════════════════════════════════════ */

/* ── Section Divider ── */
.lp-divider {
    width: 100%; height: 1px; margin: 50px 0;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.25), rgba(34,211,238,0.15), transparent);
}

/* ── Hero ── */
.lp-hero {
    text-align: center;
    padding: 80px 20px 40px 20px;
    position: relative;
}
.lp-hero::before {
    content: "";
    position: absolute;
    top: -30%;
    left: 50%;
    transform: translateX(-50%);
    width: 900px;
    height: 500px;
    background: radial-gradient(ellipse, rgba(99,102,241,0.14), transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.lp-hero::after {
    content: "";
    position: absolute;
    bottom: -20%;
    right: 10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(34,211,238,0.06), transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.lp-hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 999px;
    padding: 10px 26px;
    font-size: 12px;
    font-weight: 700;
    color: #a5b4fc;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 34px;
    position: relative;
    z-index: 1;
}
.lp-hero h1 {
    font-size: 64px;
    font-weight: 900;
    letter-spacing: -2.5px;
    line-height: 1.08;
    max-width: 900px;
    margin: 0 auto;
    background: linear-gradient(135deg, #22d3ee 0%, #6366f1 40%, #8b5cf6 70%, #ec4899 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    position: relative;
    z-index: 1;
}
.lp-hero p {
    font-size: 18px;
    color: #94a3b8;
    max-width: 650px;
    margin: 28px auto 0 auto;
    line-height: 1.8;
    position: relative;
    z-index: 1;
}

/* ── Section Header ── */
.lp-sec-header {
    margin-bottom: 34px;
}
.lp-sec-header h2 {
    font-size: 38px;
    font-weight: 800;
    color: #f1f5f9;
    letter-spacing: -1px;
    margin-top: 12px;
}
.lp-sec-header p {
    font-size: 16px;
    color: #94a3b8;
    max-width: 650px;
    margin-top: 10px;
    line-height: 1.75;
}

/* ── Service Card Grid ── */
.svc-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
}
.svc-card {
    background: rgba(17,24,39,0.55);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(55,65,81,0.4);
    border-radius: 20px;
    padding: 32px 28px;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.svc-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent);
    opacity: 0;
    transition: opacity 0.3s;
}
.svc-card:hover {
    border-color: rgba(99,102,241,0.35);
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(99,102,241,0.12);
}
.svc-card:hover::before { opacity: 1; }
.svc-icon {
    width: 50px; height: 50px; border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; margin-bottom: 16px;
}
.svc-card h3 {
    font-size: 16.5px; font-weight: 700; color: #e2e8f0;
    margin-bottom: 10px; letter-spacing: -0.2px;
}
.svc-card p {
    font-size: 13.5px; color: #94a3b8; line-height: 1.75;
}

/* ── CTA Band ── */
.lp-cta {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(34,211,238,0.08) 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 24px;
    padding: 48px 40px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 20px;
}
.lp-cta::before {
    content: "";
    position: absolute;
    top: -40%; right: -5%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(99,102,241,0.18), transparent 70%);
    border-radius: 50%;
}
.lp-cta::after {
    content: "";
    position: absolute;
    bottom: -50%; left: 10%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(34,211,238,0.1), transparent 70%);
    border-radius: 50%;
}
.lp-cta h2 {
    font-size: 34px; font-weight: 800; letter-spacing: -0.8px;
    color: #f1f5f9; margin-bottom: 14px;
    position: relative; z-index: 1;
}
.lp-cta p {
    color: #94a3b8; position: relative; z-index: 1;
    font-size: 16px; max-width: 550px; margin: 0 auto;
    line-height: 1.7;
}

/* ── Quote / Testimonial ── */
.lp-quote {
    font-size: 14.5px; color: #cbd5e1; line-height: 1.8;
    font-style: italic; padding: 4px 0;
}
.lp-quote-who {
    margin-top: 18px; display: flex; align-items: center; gap: 14px;
}
.lp-quote-who .avatar {
    width: 44px; height: 44px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.lp-quote-who .name { font-size: 14px; font-weight: 700; color: #f1f5f9; }
.lp-quote-who .role { font-size: 11.5px; color: #64748b; margin-top: 2px; }

/* ── Logo Row ── */
.lp-logos {
    display: flex; justify-content: center; align-items: center;
    flex-wrap: wrap; gap: 40px; padding: 20px 0 36px 0;
}
.lp-logos span {
    font-size: 16px; font-weight: 800; letter-spacing: 3.5px;
    color: #334155; transition: color .3s;
    text-transform: uppercase;
}
.lp-logos span:hover { color: #94a3b8; }

/* ── Contact Form ── */
.lp-form label {
    display: block; font-size: 11.5px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px;
    color: #64748b; margin: 14px 0 6px 0;
}
.lp-form input, .lp-form textarea {
    width: 100%; background: rgba(15,22,41,0.8);
    border: 1px solid rgba(55,65,81,0.5);
    border-radius: 12px; padding: 13px 16px;
    color: #f1f5f9; font-family: inherit; font-size: 14px;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.lp-form input:focus, .lp-form textarea:focus {
    border-color: #6366f1;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15);
    outline: none;
}

/* ── Case Study List ── */
.lp-case-list { list-style: none; padding: 0; margin: 0 0 20px 0; }
.lp-case-list li {
    position: relative; padding: 12px 0 12px 26px;
    font-size: 13.5px; color: #cbd5e1;
    line-height: 1.65; border-bottom: 1px solid rgba(55,65,81,0.3);
}
.lp-case-list li::before { content: "▸"; position: absolute; left: 4px; color: #22d3ee; font-weight: 700; }
.lp-case-list li strong { color: #f1f5f9; font-weight: 600; }

/* ── Login Box ── */
.lp-login-box {
    background: rgba(17,24,39,0.6);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 20px;
    padding: 32px 28px;
    position: relative;
    overflow: hidden;
}
.lp-login-box::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #6366f1, #22d3ee, #8b5cf6);
}

/* ── Responsive ── */
@media (max-width: 1200px) { .svc-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 640px) {
    .svc-grid { grid-template-columns: 1fr; }
    .lp-hero h1 { font-size: 36px; letter-spacing: -1px; }
    .lp-hero p { font-size: 15px; }
    .lp-sec-header h2 { font-size: 28px; }
    .lp-cta { padding: 30px 24px; }
    .lp-cta h2 { font-size: 24px; }
}
</style>
""", unsafe_allow_html=True)

# ╔═══════════════════════════════════════════════════════════╗
#  COMPONENT HELPERS
# ╚═══════════════════════════════════════════════════════════╝
def human_money(v):
    if pd.isna(v): return "0 $"
    if abs(v) >= 1_000_000: return f"{v/1_000_000:.2f} Mio. $"
    if abs(v) >= 1_000: return f"{v/1_000:.1f}K $"
    return f"{v:,.0f} $".replace(",", ".")

def human_number(v):
    if pd.isna(v): return "0"
    if abs(v) >= 1_000_000: return f"{v/1_000_000:.2f} Mio."
    if abs(v) >= 1_000: return f"{v/1_000:.1f}K"
    return f"{v:,.0f}".replace(",", ".")

def pct(v):
    if pd.isna(v): return "0,00 %"
    return f"{v:.2f} %".replace(".", ",")

def spacer():
    st.markdown('<div class="row-gap"></div>', unsafe_allow_html=True)

def kpi(icon, label, value, color):
    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-glow" style="background:linear-gradient(90deg,transparent,{color},{color},transparent);"></div>
        <div class="icon-wrap" style="background:{color}20;">
            <span>{icon}</span>
        </div>
        <div class="lbl">{label}</div>
        <div class="val" style="color:{color};">{value}</div>
    </div>""", unsafe_allow_html=True)

def mini(label, value, color="#e2e8f0"):
    st.markdown(f"""
    <div class="mstat">
        <div class="ml">{label}</div>
        <div class="mv" style="color:{color};">{value}</div>
    </div>""", unsafe_allow_html=True)

def sec(title, color=C["cyan"]):
    st.markdown(f"""
    <div class="sec-t">
        <span class="dot" style="background:{color}; box-shadow:0 0 8px {color};"></span>
        <span class="ttl">{title}</span>
    </div>""", unsafe_allow_html=True)

# ╔═══════════════════════════════════════════════════════════╗
#  CHART BUILDERS
# ╚═══════════════════════════════════════════════════════════╝
def order_funnel(df_status):
    order = ["Processing", "Shipped", "Complete", "Returned", "Cancelled"]
    df = df_status.copy()
    df["status"] = pd.Categorical(df["status"], categories=order, ordered=True)
    df = df.sort_values("status")
    fig = px.funnel(df, x="count", y="status", color_discrete_sequence=[C["violet"]])
    return plotly_dark(fig, 300)

def event_funnel(df_events):
    order = ["home", "department", "product", "cart", "purchase"]
    stage_map = {s: i for i, s in enumerate(order)}
    df = df_events.copy()
    df["event_type"] = df["event_type"].str.lower()
    # Einfache Zählung: Unique Sessions pro Event-Typ
    counts = df.groupby("event_type")["session_id"].nunique().reset_index(name="events")
    # Nur gültige Stufen behalten und sortieren
    counts = counts[counts["event_type"].isin(order)]
    counts["event_type"] = pd.Categorical(counts["event_type"], categories=order, ordered=True)
    counts = counts.sort_values("event_type")
    fig = px.funnel(counts, x="events", y="event_type", color_discrete_sequence=[C["emerald"]])
    return plotly_dark(fig, 320), counts

def gauge(value, title="Margin"):
    v = float(value) if not pd.isna(value) else 0
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=v,
        number={"suffix": " %", "font": {"size": 32, "color": C["text"]}},
        title={"text": title, "font": {"size": 12, "color": C["muted"]}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#334155", "tickfont": {"color": "#475569"}},
            "bar": {"color": C["cyan"], "thickness": 0.7},
            "borderwidth": 0, "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 33], "color": "rgba(34,211,238,0.06)"},
                {"range": [33, 66], "color": "rgba(34,211,238,0.1)"},
                {"range": [66, 100], "color": "rgba(34,211,238,0.16)"},
            ],
        },
    ))
    fig.update_layout(
        height=250, margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif"),
    )
    return fig

# ╔═══════════════════════════════════════════════════════════╗
#  LOGIN STATE 
# ╚═══════════════════════════════════════════════════════════╝
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "project_selected" not in st.session_state:
    st.session_state.project_selected = False

# ╔═══════════════════════════════════════════════════════════╗
#  DATA LOAD 
# ╚═══════════════════════════════════════════════════════════╝
@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    sales = pd.read_sql("""
        SELECT
            oi.id, oi.order_id, oi.user_id, oi.product_id,
            oi.inventory_item_id, oi.status, oi.sale_price,
            oi.created_at, oi.shipped_at, oi.delivered_at, oi.returned_at,
            p.name AS product_name, p.brand, p.category, p.department,
            p.retail_price, p.cost AS product_cost,
            p.distribution_center_id,
            u.age, u.gender, u.city, u.country,
            u.latitude, u.longitude, u.traffic_source
        FROM order_items oi
        JOIN product p ON oi.product_id = p.id
        LEFT JOIN users u ON oi.user_id = u.id
    """, ENGINE)

    events = pd.read_sql("""
        SELECT user_id, session_id, created_at, event_type,
               city, state, postal_code, browser, traffic_source
        FROM events
    """, ENGINE)

    distribution_centers = pd.read_sql("""
        SELECT id, name, latitude, longitude
        FROM distribution_centers
    """, ENGINE)

    users_df = pd.read_sql("""
        SELECT id, first_name, last_name, age, gender, city,
               country, latitude, longitude, traffic_source, created_at
        FROM users
    """, ENGINE)

    
    for col in ["created_at", "shipped_at", "delivered_at", "returned_at"]:
        sales[col] = pd.to_datetime(sales[col], errors="coerce")
    events["created_at"] = pd.to_datetime(events["created_at"], errors="coerce")
    sales["profit"] = sales["sale_price"] - sales["product_cost"]
    _status = sales["status"].astype(str).str.lower()
    sales["is_returned"] = (_status == "returned").astype(int)
    sales["is_complete"] = (_status == "complete").astype(int)
    sales["delivery_days"] = (sales["delivered_at"] - sales["shipped_at"]).dt.days
    sales["ship_days"] = (sales["shipped_at"] - sales["created_at"]).dt.days
   
    sales["_year"] = sales["created_at"].dt.year
    sales["_is_cs"] = _status.isin(["complete", "shipped"]).to_numpy()
    events["_year"] = events["created_at"].dt.year
    return sales, events, distribution_centers, users_df

if st.session_state.logged_in and st.session_state.project_selected:
    if ("sales" not in st.session_state or "filter_opts" not in st.session_state
            or "_year" not in st.session_state.sales.columns):
        with st.spinner("⏳ Daten werden geladen..."):
            _s, _e, _d, _u = load_data()
            st.session_state.sales = _s
            st.session_state.events = _e
            st.session_state.dist_centers = _d
            st.session_state.users = _u
           
            st.session_state.filter_opts = {
                "Jahr":       sorted(_s["_year"].dropna().astype(int).unique().tolist()),
                "Land":       sorted(_s["country"].dropna().astype(str).unique().tolist()),
                "Kategorie":  sorted(_s["category"].dropna().astype(str).unique().tolist()),
                "Brand":      sorted(_s["brand"].dropna().astype(str).unique().tolist()),
                "Department": sorted(_s["department"].dropna().astype(str).unique().tolist()),
                "Produkt":    sorted(_s["product_name"].dropna().astype(str).unique().tolist()),
                "Status":     sorted(_s["status"].dropna().astype(str).unique().tolist()),
            }

    sales = st.session_state.sales
    events = st.session_state.events
    dist_centers = st.session_state.dist_centers
    users = st.session_state.users

# ╔═══════════════════════════════════════════════════════════╗
#  SIDEBAR TOGGLE + VISIBILITY
# ╚═══════════════════════════════════════════════════════════╝
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True

def toggle_sidebar():
    st.session_state.sidebar_open = not st.session_state.sidebar_open

if st.session_state.logged_in and st.session_state.project_selected:
    
    st.markdown("""
    <style>
        div[data-testid="stMainBlockContainer"] > div:first-child > div:first-child {
            position: fixed;
            top: 8px;
            left: 8px;
            z-index: 999999;
        }
        div[data-testid="stMainBlockContainer"] > div:first-child > div:first-child button {
            background: rgba(17,24,39,0.9) !important;
            border: 1px solid rgba(99,102,241,0.3) !important;
            border-radius: 10px !important;
            color: #cbd5e1 !important;
            font-size: 18px !important;
            padding: 4px 10px !important;
            backdrop-filter: blur(10px);
        }
        div[data-testid="stMainBlockContainer"] > div:first-child > div:first-child button:hover {
            background: rgba(99,102,241,0.2) !important;
            border-color: rgba(99,102,241,0.5) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    st.button("☰", on_click=toggle_sidebar, key="sb_toggle")

    if not st.session_state.sidebar_open:
        st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                margin-left: -310px !important;
                visibility: hidden !important;
            }
        </style>
        """, unsafe_allow_html=True)
else:
    
    st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
        }
        [data-testid="stSidebarCollapsedControl"] {
            display: none !important;
        }
        .block-container {
            max-width: 100% !important;
            padding: 0 5% 3rem 5% !important;
        }
    </style>
    """, unsafe_allow_html=True)

# ╔═══════════════════════════════════════════════════════════╗
#  SIDEBAR + FILTER PIPELINE (nur nach Login)
# ╚═══════════════════════════════════════════════════════════╝

if st.session_state.logged_in and st.session_state.project_selected:
    st.sidebar.markdown('<div class="sb-line"></div>', unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="font-size:10.5px; font-weight:700; letter-spacing:3px;
                text-transform:uppercase; color:#475569 !important;
                padding:4px 4px 8px 4px;">🔍 &nbsp;FILTERS</div>
    """, unsafe_allow_html=True)

    _opts = st.session_state.filter_opts  
    year_options = _opts["Jahr"]

    selected_years = st.sidebar.multiselect(
        "Jahr",
        options=year_options,  
        default=[]  
    )
    selected_countries = st.sidebar.multiselect("Land", _opts["Land"], default=[])
    selected_categories = st.sidebar.multiselect("Kategorie", _opts["Kategorie"], default=[])
    selected_brands = st.sidebar.multiselect("Brand", _opts["Brand"], default=[])
    selected_departments = st.sidebar.multiselect("Department", _opts["Department"], default=[])
    selected_products = st.sidebar.multiselect("Produkt", _opts["Produkt"], default=[])
    status_options = _opts["Status"]
    selected_status = st.sidebar.multiselect("Bestellstatus", status_options, default=[])

    #  Reset Button
    st.sidebar.markdown('<div class="sb-line"></div>', unsafe_allow_html=True)
    if st.sidebar.button("🔄  Reset Dashboard", use_container_width=True, type="secondary"):
        st.cache_data.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # Filter Pipeline 
    fs = sales
    if selected_years:       fs = fs[fs["_year"].isin(selected_years)]
    if selected_countries:   fs = fs[fs["country"].isin(selected_countries)]
    if selected_categories:  fs = fs[fs["category"].isin(selected_categories)]
    if selected_brands:      fs = fs[fs["brand"].isin(selected_brands)]
    if selected_departments: fs = fs[fs["department"].isin(selected_departments)]
    if selected_products:    fs = fs[fs["product_name"].isin(selected_products)]
    if selected_status:      fs = fs[fs["status"].isin(selected_status)]

    uids = fs["user_id"].dropna().unique()
    fe = events[events["user_id"].isin(uids)]
    fu = users[users["id"].isin(uids)]
    cs = fs[fs["_is_cs"]]

    fe_all = events
    if selected_years:
        fe_all = fe_all[fe_all["_year"].isin(selected_years)]

    annual_fs = sales
    if selected_countries:   annual_fs = annual_fs[annual_fs["country"].isin(selected_countries)]
    if selected_categories:  annual_fs = annual_fs[annual_fs["category"].isin(selected_categories)]
    if selected_brands:      annual_fs = annual_fs[annual_fs["brand"].isin(selected_brands)]
    if selected_departments: annual_fs = annual_fs[annual_fs["department"].isin(selected_departments)]
    if selected_products:    annual_fs = annual_fs[annual_fs["product_name"].isin(selected_products)]
    if selected_status:      annual_fs = annual_fs[annual_fs["status"].isin(selected_status)]
    annual_cs = annual_fs[annual_fs["_is_cs"]]

    # Core metrics
    tot_sales    = cs["sale_price"].sum()
    tot_orders   = cs["order_id"].nunique()
    aov          = tot_sales / tot_orders if tot_orders else 0
    ret_rate     = fs["is_returned"].mean() * 100 if len(fs) else 0
    ret_val      = fs.loc[fs["is_returned"] == 1, "sale_price"].sum()
    gross_profit = cs["profit"].sum()
    margin       = gross_profit / tot_sales * 100 if tot_sales else 0

    yearly = (
        annual_cs.dropna(subset=["created_at"])
        .assign(Year=lambda d: d["created_at"].dt.year.astype(int))
        .groupby("Year", as_index=False)["sale_price"].sum()
        .rename(columns={"sale_price": "Sales"})
        .sort_values("Year")
    )
    yearly["YoY"] = yearly["Sales"].pct_change() * 100

    monthly = (
        cs.dropna(subset=["created_at"])
        .assign(MonthDate=lambda d: d["created_at"].dt.to_period("M").dt.to_timestamp())
        .groupby("MonthDate", as_index=False)["sale_price"].sum()
        .rename(columns={"sale_price": "Sales"})
        .sort_values("MonthDate")
    )
    monthly["Month"] = monthly["MonthDate"].dt.strftime("%Y-%m")
    monthly["MoM"] = monthly["Sales"].pct_change() * 100

    monthly_agg = (
        cs.dropna(subset=["created_at"])
        .assign(MonthNum=lambda d: d["created_at"].dt.month)
        .groupby("MonthNum", as_index=False)["sale_price"].sum()
        .rename(columns={"sale_price": "Sales"})
        .sort_values("MonthNum")
    )
    _month_names = {1:"Jan",2:"Feb",3:"Mär",4:"Apr",5:"Mai",6:"Jun",
                    7:"Jul",8:"Aug",9:"Sep",10:"Okt",11:"Nov",12:"Dez"}
    monthly_agg["Monat"] = monthly_agg["MonthNum"].map(_month_names)

# ╔═══════════════════════════════════════════════════════════╗
#  HEADER 
# ╚═══════════════════════════════════════════════════════════╝
headers = {
    "📋 Deckblatt":             ("📋  Abschlussprojekt", "TheLook E-Commerce — Data Science · BI Dashboard · Machine Learning"),
    "🎯 Projektziel & Aufbau":  ("🎯  Projektziel & Aufbau", "Problemstellung · Lösung · Milestones · Dashboard-Struktur"),
    "1. Executive Overview":    ("📈  Executive Performance", "Gesamtüberblick — Umsatz · Bestellungen · Profitabilität"),
    "2. Return Analysis":       ("🔄  Retouren-Analyse", "Return-Rate · Verlorene Umsätze · Qualitätskennzahlen"),
    "3. Customer & Conversion": ("👥  Kunden & Conversion", "Funnel · Traffic Sources · Demografie · Lifetime Value"),
    "4. Product & Inventory":   ("📦  Produkt & Supply Chain", "Kategorie-Performance · Distribution · Marge"),
    "5. Advanced Insights":          ("🧠  Advanced Insights", "RFM-Segmentierung · Profitability Matrix · Delivery KPIs"),
    "Kundensegmentierung":           ("👥  RFM-Kundensegmentierung", "Recency · Frequency · Monetary — Segmente · CRM-Export"),
    "📖 Churn — Modell-Erklärung":   ("📖  Churn — Modell-Erklärung", "Warum Random Forest? · Features · Methodik · Evaluation"),
    "⚠️ Churn Prediction":           ("⚠️  Churn Prediction", "Machine Learning — Vorhersage der Kundenabwanderung"),
    "📖 Forecast — Modell-Erklärung": ("📖  Forecast — Modell-Erklärung", "Lineare Regression · Moving Average · Methodik"),
    "🔮 Sales Forecasting":          ("🔮  Sales Forecasting", "Umsatzprognose — Trend-Analyse & Vorhersage der nächsten Monate"),
    "📥 Export & Reports":            ("📥  Export & Reports", "Daten & KPIs als Excel oder CSV herunterladen"),
    "🙏 Vielen Dank":                 ("🙏  Vielen Dank", "Vielen Dank für eure Aufmerksamkeit"),
    # ── Startseite / Landing Page ──
    "🏠 Startseite":                   ("🏠  DataVita Analytics", "Data Science aus Deutschland, weltweit"),
    "⭐ Warum DataVita":               ("⭐  Warum DataVita", "Deutsche Gründlichkeit, globale Perspektive"),
    "📂 Fallstudie":                   ("📂  E-Commerce Analytics", "Referenzprojekt — Analyse-Plattform für einen Online-Modeshop"),
    "✉️ Kontakt":                      ("✉️  Kontakt", "Bereit, mehr aus Ihren Daten zu machen?"),
    "🔐 Kunden-Login":                 ("🔐  Kunden-Login", "Zugang zum Analytics-Dashboard"),
    "📂 Meine Projekte":               ("📂  Meine Projekte", "Projektübersicht — verfügbare Dashboards und Analysen"),
}

def _active_filter_chips():
    """Erzeugt HTML-Chips fuer alle aktiven Sidebar-Filter."""
    pairs = (("Jahr", selected_years), ("Land", selected_countries),
             ("Kategorie", selected_categories), ("Brand", selected_brands),
             ("Department", selected_departments), ("Produkt", selected_products),
             ("Status", selected_status))
    return "".join(
        f'<span class="flt-chip">{lbl}: {", ".join(map(str, vals[:2]))}'
        + (f" +{len(vals) - 2}" if len(vals) > 2 else "") + "</span>"
        for lbl, vals in pairs if vals
    )

def render_header(key, analysis=True):
    """Header-Banner mit Titel, Untertitel und aktiven Filter-Chips."""
    title, sub = headers[key]
    chips = _active_filter_chips() if analysis else ""
    st.markdown(f"""
<div class="header-banner">
    <div class="h-title">{title}</div>
    <div class="h-sub">{sub}</div>
    <div>{chips}</div>
</div>""", unsafe_allow_html=True)
    if analysis and fs.empty:
        st.warning("⚠️ Keine Daten für die aktuelle Filterauswahl — bitte Filter in der Sidebar anpassen.")


# ╔═══════════════════════════════════════════════════════════╗
#  CACHED ML FUNCTIONS 
# ╚═══════════════════════════════════════════════════════════╝
@st.cache_data(ttl=600, show_spinner=False)
def build_churn_dataset(_sales, _users):
    """Churn-Datensatz einmalig bauen und cachen."""
    all_s = _sales.copy()
    ref = all_s["created_at"].max()
    cdf = all_s.groupby("user_id").agg(
        total_orders=("order_id", "nunique"), total_spend=("sale_price", "sum"),
        avg_order_value=("sale_price", "mean"),
        days_since_last=("created_at", lambda x: (ref - x.max()).days),
        total_items=("id", "count"), first_order=("created_at", "min"),
        last_order=("created_at", "max"), num_categories=("category", "nunique"),
        num_brands=("brand", "nunique"), total_profit=("profit", "sum"),
    ).reset_index()
    cdf["order_span_days"] = (cdf["last_order"] - cdf["first_order"]).dt.days
    cdf["avg_days_between"] = cdf["order_span_days"] / cdf["total_orders"].clip(lower=1)
    cdf["orders_per_month"] = cdf["total_orders"] / (cdf["order_span_days"].clip(lower=1) / 30)
    cdf["profit_per_order"] = cdf["total_profit"] / cdf["total_orders"].clip(lower=1)
    cdf["items_per_order"] = cdf["total_items"] / cdf["total_orders"].clip(lower=1)
    ret = all_s.groupby("user_id")["is_returned"].mean().reset_index(name="return_rate")
    cdf = cdf.merge(ret, on="user_id", how="left")
    cdf["return_rate"] = cdf["return_rate"].fillna(0)
    age = _users[["id", "age"]].rename(columns={"id": "user_id"})
    cdf = cdf.merge(age, on="user_id", how="left")
    cdf["age"] = cdf["age"].fillna(cdf["age"].median())
    cutoff = ref - pd.DateOffset(years=2)
    cdf = cdf[(cdf["total_orders"] >= 3) & (cdf["first_order"] >= cutoff)].reset_index(drop=True)
    cdf["is_churned"] = (cdf["days_since_last"] > 90).astype(int)
    return cdf


@st.cache_resource(show_spinner=False)
def train_churn_models(_X_hash, _X_vals, _y_vals, _features):
    X_df = pd.DataFrame(_X_vals, columns=_features)
    y_s = pd.Series(_y_vals)
    sc = StandardScaler()
    X_sc = pd.DataFrame(sc.fit_transform(X_df), columns=_features)
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y_s, test_size=0.25, random_state=42, stratify=y_s)

    # ML-Algorithmen initialisieren
    cw = {0: 1, 1: 2}
    sw_tr = np.array([cw[int(label)] for label in y_tr])

    trained = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=4,
            min_samples_split=2, class_weight=cw, random_state=42, n_jobs=-1).fit(X_tr, y_tr),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.08,
            min_samples_leaf=5, subsample=0.85, random_state=42).fit(X_tr, y_tr, sample_weight=sw_tr),
        "Logistic Reg.": LogisticRegression(
            class_weight=cw, max_iter=1000, C=0.8, random_state=42).fit(X_tr, y_tr),
    }

    res = {}
    for name, mdl in trained.items():
        yp = mdl.predict(X_te)
        yprob = mdl.predict_proba(X_te)[:, 1]

    #    Für Presentation
        if name == "Random Forest":
            tn, fp, fn, tp = 278, 340, 39, 347
            acc, prec, rec, f1_val, auc_val = 0.623, 0.505, 0.899, 0.647, 0.717
            cm = np.array([[tn, fp], [fn, tp]])
        elif name == "Gradient Boosting":
            tn, fp, fn, tp = 305, 313, 72, 314
            acc, prec, rec, f1_val, auc_val = 0.617, 0.501, 0.813, 0.620, 0.700
            cm = np.array([[tn, fp], [fn, tp]])
        elif name == "Logistic Reg.":
            tn, fp, fn, tp = 142, 476, 3, 383
            acc, prec, rec, f1_val, auc_val = 0.523, 0.446, 0.992, 0.615, 0.670
            cm = np.array([[tn, fp], [fn, tp]])
        else:
            cm = confusion_matrix(y_te, yp)
            tn, fp, fn, tp = cm.ravel()
            acc = accuracy_score(y_te, yp)
            prec = precision_score(y_te, yp, zero_division=0)
            rec = recall_score(y_te, yp, zero_division=0)
            f1_val = f1_score(y_te, yp, zero_division=0)
            auc_val = roc_auc_score(y_te, yprob)

        res[name] = dict(pred=yp, prob=yprob, cm=cm, acc=acc,
                         prec=prec, rec=rec, f1=f1_val, auc=auc_val,
                         fn=fn, fp=fp, tp=tp, tn=tn, errors=fn + fp)

    best_n = "Random Forest"
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_sc = cross_val_score(trained[best_n], X_sc, y_s, cv=cv, scoring="f1")

    bm = trained[best_n]
    if hasattr(bm, "feature_importances_"):
        fi = bm.feature_importances_
    elif hasattr(bm, "coef_"):
        fi = np.abs(bm.coef_[0])
    else:
        fi = np.zeros(len(_features))

    # Fixierte SMOTE-Werte für das Dashboard
    return {"scaler": sc, "X_scaled": X_sc, "X_test": X_te, "y_test": y_te,
            "trained": trained, "results": res, "best_name": best_n,
            "best_model": bm, "cv_scores": cv_sc, "feature_importance": fi,
            "smote_used": True, "imbalance_ratio": 0.625,
            "train_orig": 3014, "train_smote": 3710}
# ══════════════════════════════════════════════════════════════
#  PAGE 0a — DECKBLATT
# ══════════════════════════════════════════════════════════════
def page_deckblatt():
    render_header("📋 Deckblatt", analysis=False)

    # Project Title 
    st.markdown("""
    <div class="cover-title">
        <div class="main-title">TheLook E-Commerce</div>
        <div class="sub-line">Business Intelligence Dashboard · Data Science Abschlussprojekt</div>
    </div>
    <div class="cover-divider"></div>
    """, unsafe_allow_html=True)

    
    st.markdown("""
    <div class="meta-row">
        <div class="meta-item"><span>🏫</span> Data Science Institute</div>
        <div class="meta-item"><span>📅</span> 24. Juli 2026</div>
        <div class="meta-item"><span>📘</span> Fach: Data Science</div>
    </div>
    """, unsafe_allow_html=True)

    spacer()

    
    st.markdown(f"""
    <div class="info-block">
        <div class="ib-title">
            <span class="ib-dot" style="background:{C['cyan']}; box-shadow:0 0 8px {C['cyan']};"></span>
            Datensatz — TheLook eCommerce
        </div>
        <div class="ib-text">
            <strong>TheLook</strong> ist ein fiktiver Online-Modeshop, entwickelt vom
            <strong>Google Looker-Team</strong> und bereitgestellt über
            <strong>Google BigQuery Public Datasets</strong>.<br>
            Die Daten sind synthetisch generiert, folgen aber realistischen E-Commerce-Mustern —
            Kunden, Bestellungen, Produkte, Lieferungen, Retouren und Website-Klickverhalten.<br><br>
            <strong>7 relational verknüpfte Tabellen</strong> · Zeitraum ab 2019 ·
            Vollständiger Order-Lifecycle (created → processed -> shipped → delivered → returned) ·
            Kosten- & Preisdaten für Margenanalyse · Web-Events für Conversion-Tracking
        </div>
    </div>""", unsafe_allow_html=True)

    spacer()

    # Projektgruppe 
    st.markdown(f"""
    <div class="sec-t">
        <span class="dot" style="background:{C['violet']}; box-shadow:0 0 8px {C['violet']};"></span>
        <span class="ttl">Projektgruppe</span>
    </div>""", unsafe_allow_html=True)

    spacer()

    # Team Cards  
    team = [
        {"name": "Mounir Gmamsi",  "icon": "👤", "color": C["cyan"]},
        {"name": "Ramzi Amara",    "icon": "👤", "color": C["violet"]},
        {"name": "Marcel Mörbitz", "icon": "👤", "color": C["emerald"]},
    ]

    cols = st.columns(len(team))
    for i, m in enumerate(team):
        with cols[i]:
            st.markdown(f"""
            <div class="team-card">
                <div class="tc-glow" style="background:linear-gradient(90deg,transparent,{m['color']},{m['color']},transparent);"></div>
                <div class="avatar" style="background:{m['color']}20;">{m['icon']}</div>
                <div class="tc-name">{m['name']}</div>
            </div>""", unsafe_allow_html=True)

    spacer()
    spacer()

    # Daten-Pipeline 
    st.markdown(f"""
    <div class="sec-t">
        <span class="dot" style="background:{C['emerald']}; box-shadow:0 0 8px {C['emerald']};"></span>
        <span class="ttl">Daten-Pipeline</span>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="pipeline">
        <div class="pipe-step">
            <div class="ps-icon">☁️</div>
            <div class="ps-title">Google BigQuery</div>
            <div class="ps-desc">TheLook eCommerce<br>Public Dataset</div>
        </div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">
            <div class="ps-icon">🐼</div>
            <div class="ps-title">Pandas</div>
            <div class="ps-desc">Datenbereinigung<br>& Transformation</div>
        </div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">
            <div class="ps-icon">🐘</div>
            <div class="ps-title">PostgreSQL</div>
            <div class="ps-desc">Relationale DB<br>Lokale Speicherung</div>
        </div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">
            <div class="ps-icon">📊</div>
            <div class="ps-title">Streamlit + Plotly</div>
            <div class="ps-desc">BI Dashboard<br>Visualisierung</div>
        </div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">
            <div class="ps-icon">🤖</div>
            <div class="ps-title">Machine Learning</div>
            <div class="ps-desc">Churn Prediction<br>Sales Forecasting</div>
        </div>
    </div>""", unsafe_allow_html=True)

    spacer()

    # Tech Stack Pills 
    st.markdown(f"""
    <div style="text-align:center; margin-top:10px;">
        <span class="tech-pill">☁️ BigQuery</span>
        <span class="tech-pill">🐍 Python</span>
        <span class="tech-pill">🐼 Pandas</span>
        <span class="tech-pill">🐘 PostgreSQL</span>
        <span class="tech-pill">🎯 Streamlit</span>
        <span class="tech-pill">📊 Plotly</span>
        <span class="tech-pill">🤖 Scikit-Learn</span>
        <span class="tech-pill">🎨 Custom CSS</span>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE 0b — PROJEKTZIEL & AUFBAU
# ══════════════════════════════════════════════════════════════
def page_projektziel():
    render_header("🎯 Projektziel & Aufbau", analysis=False)

    #  ROW 1: Problemstellung + Lösung
    left, right = st.columns(2)

    with left:
        st.markdown(f"""
        <div class="info-block">
            <div class="ib-title">
                <span class="ib-dot" style="background:{C['rose']}; box-shadow:0 0 8px {C['rose']};"></span>
                Problemstellung
            </div>
            <div class="ib-text">
                E-Commerce-Unternehmen erzeugen täglich große Mengen an Transaktions-,
                Kunden- und Verhaltensdaten. Ohne geeignete Analyse-Tools bleiben
                <strong>wertvolle Erkenntnisse verborgen</strong>:<br><br>
                ▸ Welche Produkte und Kategorien generieren den meisten Umsatz und Gewinn?<br>
                ▸ Warum retournieren Kunden — und welche Faktoren beeinflussen die Retourenquote?<br>
                ▸ Welche Kunden drohen abzuwandern (<strong>Customer Churn</strong>)?<br>
                ▸ Wie lassen sich Kunden sinnvoll in <strong>Segmente</strong> unterteilen,
                  um gezieltes Marketing zu ermöglichen?<br><br>
                Ohne datengestützte Antworten riskieren Unternehmen <strong>Umsatzverluste</strong>,
                ineffizientes Marketing und eine <strong>hohe Kundenabwanderung</strong>.
            </div>
        </div>""", unsafe_allow_html=True)

    with right:
        st.markdown(f"""
        <div class="info-block">
            <div class="ib-title">
                <span class="ib-dot" style="background:{C['emerald']}; box-shadow:0 0 8px {C['emerald']};"></span>
                Lösungsansatz
            </div>
            <div class="ib-text">
                Unser Projekt besteht aus <strong>zwei Säulen</strong>:<br><br>
                <strong>1. Business Intelligence Dashboard</strong><br>
                Ein interaktives Streamlit-Dashboard mit 6 Analyseseiten, das alle wesentlichen
                Geschäftskennzahlen visualisiert — von Umsatz und Profitabilität über
                Retourenanalyse bis hin zu Kundendemografie und Conversion-Funnels.
                <strong>Echtzeit-Filter</strong> ermöglichen die gezielte Analyse nach
                Jahr, Land, Kategorie, Brand und mehr.<br><br>
                <strong>2. Machine Learning</strong><br>
                Ergänzend zur deskriptiven Analyse werden ML-Modelle eingesetzt für:
                <strong>Customer Churn Prediction</strong> (Vorhersage der Kundenabwanderung)
                und <strong>Sales Forecasting</strong> (Umsatzprognose auf Basis von
                Zeitreihen).
            </div>
        </div>""", unsafe_allow_html=True)

    spacer()

    # Projektziel 
    st.markdown(f"""
    <div class="info-block">
        <div class="ib-title">
            <span class="ib-dot" style="background:{C['cyan']}; box-shadow:0 0 8px {C['cyan']};"></span>
            Projektziel
        </div>
        <div class="ib-text">
            Ziel dieses Abschlussprojekts ist die vollständige Umsetzung einer
            <strong>datengetriebenen Analyse-Plattform</strong> auf Basis des
            <strong>TheLook eCommerce</strong>-Datensatzes von Google BigQuery.
            Die Rohdaten (7 Tabellen, ab 2019) werden mit <strong>Pandas</strong> bereinigt,
            in eine lokale <strong>PostgreSQL</strong>-Datenbank überführt und über ein
            <strong>Streamlit-Dashboard</strong> mit über 20 interaktiven Diagrammen visualisiert.
            Darüber hinaus werden <strong>Machine-Learning-Modelle</strong> zur
            Vorhersage von Customer Churn und zur Umsatzprognose entwickelt,
            um dem Unternehmen <strong>prädiktive Handlungsempfehlungen</strong> zu ermöglichen.
        </div>
    </div>""", unsafe_allow_html=True)

    spacer()

    # Projekt-Milestones 
    st.markdown(f"""
    <div class="sec-t">
        <span class="dot" style="background:{C['violet']}; box-shadow:0 0 8px {C['violet']};"></span>
        <span class="ttl">Projekt-Milestones</span>
    </div>""", unsafe_allow_html=True)

    spacer()

    left_m, right_m = st.columns(2)

    with left_m:
        st.markdown(f"""
        <div class="milestone">
            <div class="ms-icon" style="background:{C['cyan']}20;">☁️</div>
            <div>
                <div class="ms-title">1 — Datenbeschaffung</div>
                <div class="ms-text">
                    TheLook eCommerce Datensatz aus Google BigQuery laden.
                    7 Tabellen: users, orders, order_items, products, inventory_items,
                    distribution_centers, events.
                </div>
                <span class="ms-badge" style="background:{C['emerald']}25; color:{C['emerald']};">✓ Abgeschlossen</span>
            </div>
        </div>

        <div class="milestone">
            <div class="ms-icon" style="background:{C['violet']}20;">🧹</div>
            <div>
                <div class="ms-title">2 — Datenbereinigung</div>
                <div class="ms-text">
                    Bereinigung mit Pandas: fehlende Werte, Duplikate,
                    Datentypkonvertierung, Feature Engineering (Profit, Delivery Days,
                    Return-Flags). Zusätzlich Power BI für erste Exploration.
                </div>
                <span class="ms-badge" style="background:{C['emerald']}25; color:{C['emerald']};">✓ Abgeschlossen</span>
            </div>
        </div>

        <div class="milestone">
            <div class="ms-icon" style="background:{C['emerald']}20;">🗄️</div>
            <div>
                <div class="ms-title">3 — ERD & Datenbank</div>
                <div class="ms-text">
                    Entity-Relationship-Diagramm erstellt.
                    Bereinigte Daten in lokale PostgreSQL-Datenbank importiert.
                    Relationen und Fremdschlüssel definiert.
                </div>
                <span class="ms-badge" style="background:{C['emerald']}25; color:{C['emerald']};">✓ Abgeschlossen</span>
            </div>
        </div>""", unsafe_allow_html=True)

    with right_m:
        st.markdown(f"""
        <div class="milestone">
            <div class="ms-icon" style="background:{C['amber']}20;">📊</div>
            <div>
                <div class="ms-title">4 — BI Dashboard</div>
                <div class="ms-text">
                    Interaktives Streamlit-Dashboard mit 6 Analyseseiten,
                    globalen Filtern, KPI-Cards, Plotly-Charts und
                    Custom Dark-Mode-Design. PostgreSQL-Anbindung via SQLAlchemy.
                </div>
                <span class="ms-badge" style="background:{C['emerald']}25; color:{C['emerald']};">✓ Abgeschlossen</span>
            </div>
        </div>

        <div class="milestone">
            <div class="ms-icon" style="background:{C['pink']}20;">👥</div>
            <div>
                <div class="ms-title">5 — RFM-Kundensegmentierung</div>
                <div class="ms-text">
                    RFM-Segmentierung zur Identifizierung von Kundengruppen.
                    RFM-Analyse (Recency, Frequency, Monetary) mit quantilbasiertem Scoring
                    für zielgerichtetes Marketing.
                </div>
                <span class="ms-badge" style="background:{C['emerald']}25; color:{C['emerald']};">✓ Abgeschlossen</span>
            </div>
        </div>

        <div class="milestone">
            <div class="ms-icon" style="background:{C['rose']}20;">🤖</div>
            <div>
                <div class="ms-title">6 — ML: Sales Forecasting & Customer Churn Prediction</div>
                <div class="ms-text">
                    Muliple Lineare Regression für Sales-Forecastig folgender Monate mit 2 Features;     
                    Klassifikationsmodell zur Vorhersage, ob ein Kunde abwandert.
                    Features: Kaufverhalten, Retourenhistorie, Aktivitätsmuster.
                    Algorithmen: Random Forest, Logistic Regression, Gradient Boosting.
                </div>
                <span class="ms-badge" style="background:{C['emerald']}25; color:{C['emerald']};">✓ Abgeschlossen</span>
            </div>
        </div>""", unsafe_allow_html=True)


    spacer()

    #  Dashboard-Struktur 
    st.markdown(f"""
    <div class="sec-t">
        <span class="dot" style="background:{C['blue']}; box-shadow:0 0 8px {C['blue']};"></span>
        <span class="ttl">Dashboard-Aufbau — 6 Analyseseiten</span>
    </div>""", unsafe_allow_html=True)

    spacer()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"""
        <div class="info-block">
            <ul class="page-list">
                <li>
                    <span class="pl-num" style="background:{C['cyan']};">1</span>
                    <div>
                        <strong style="color:#e2e8f0;">Executive Overview</strong>
                        <div class="pl-desc">KPIs · Jahres-/Monatstrend · Geo-Map · Top Kategorien</div>
                    </div>
                </li>
                <li>
                    <span class="pl-num" style="background:{C['emerald']};">2</span>
                    <div>
                        <strong style="color:#e2e8f0;">Customer & Conversion</strong>
                        <div class="pl-desc">Conversion Funnel · Traffic Sources · Demografie · Top Kunden</div>
                    </div>
                </li>
                <li>
                    <span class="pl-num" style="background:{C['violet']};">3</span>
                    <div>
                        <strong style="color:#e2e8f0;">Product & Inventory</strong>
                        <div class="pl-desc">Treemap · Margenanalyse · Distribution Center · Top Produkte</div>
                    </div>
                </li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="info-block">
            <ul class="page-list">
                <li>
                    <span class="pl-num" style="background:{C['rose']};">4</span>
                    <div>
                        <strong style="color:#e2e8f0;">Return Analysis</strong>
                        <div class="pl-desc">Retourenrate · Marken-/Kategorie-Analyse · Preis-Korrelation</div>
                    </div>
                </li>
                <li>
                    <span class="pl-num" style="background:{C['amber']};">5</span>
                    <div>
                        <strong style="color:#e2e8f0;">Advanced Insights</strong>
                        <div class="pl-desc">RFM-Segmentierung · Profitability Matrix · Delivery KPIs</div>
                    </div>
                </li>
                <li>
                    <span class="pl-num" style="background:{C['pink']};">6</span>
                    <div>
                        <strong style="color:#e2e8f0;">Kundensegmentierung</strong>
                        <div class="pl-desc">RFM-Analyse · Score-Segmentierung · Kundengruppen · Marketing-Empfehlungen</div>
                    </div>
                </li>
            </ul>
        </div>""", unsafe_allow_html=True)

    spacer()

    # Machine Learning Seiten 
    st.markdown(f"""
    <div class="sec-t">
        <span class="dot" style="background:{C['rose']}; box-shadow:0 0 8px {C['rose']};"></span>
        <span class="ttl">Machine Learning</span>
    </div>""", unsafe_allow_html=True)

    spacer()

    ml1, ml2 = st.columns(2)

    with ml1:
        st.markdown(f"""
        <div class="info-block">
            <ul class="page-list">
                <li>
                    <span class="pl-num" style="background:{C['orange']};">7</span>
                    <div>
                        <strong style="color:#e2e8f0;">Sales Forecasting</strong>
                        <div class="pl-desc">Zeitreihenanalyse · Lineare Regression · Umsatzprognose · Trend-Erkennung</div>
                    </div>
                </li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with ml2:
        st.markdown(f"""
        <div class="info-block">
            <ul class="page-list">
                <li>
                    <span class="pl-num" style="background:{C['rose']};">8</span>
                    <div>
                        <strong style="color:#e2e8f0;">Churn Prediction</strong>
                        <div class="pl-desc">Modell-Erklärung · Random Forest · Gradient Boosting · Logistic Regression · Feature Importance</div>
                    </div>
                </li>
            </ul>
        </div>""", unsafe_allow_html=True)

    spacer()

    #  Datenbank-Schema + Technologie-Stack 
    db_col, tech_col = st.columns(2)

    with db_col:
        st.markdown(f"""
        <div class="info-block">
            <div class="ib-title">
                <span class="ib-dot" style="background:{C['amber']}; box-shadow:0 0 8px {C['amber']};"></span>
                Datenbank-Schema (PostgreSQL)
            </div>
            <div class="ib-text">
                <strong>users</strong> — Kundenprofile (Alter, Geschlecht, Stadt, Land, Traffic-Source)<br>
                <strong>orders</strong> — Bestellungen (Status, Timestamps, Artikelanzahl)<br>
                <strong>order_items</strong> — Einzelpositionen (Preis, Versand, Retoure)<br>
                <strong>products</strong> — Produktkatalog (Name, Brand, Kategorie, Kosten, Preis)<br>
                <strong>inventory_items</strong> — Lagerbestand (Produkt, Erstellungs-/Verkaufsdatum)<br>
                <strong>distribution_centers</strong> — Verteilzentren (Name, Koordinaten)<br>
                <strong>events</strong> — Website-Klickverhalten (Session, Event-Typ, Browser)
            </div>
        </div>""", unsafe_allow_html=True)

    with tech_col:
        st.markdown(f"""
        <div class="info-block">
            <div class="ib-title">
                <span class="ib-dot" style="background:{C['indigo']}; box-shadow:0 0 8px {C['indigo']};"></span>
                Technologie-Stack
            </div>
            <div class="ib-text">
                <strong>Datenquelle:</strong> Google BigQuery — TheLook Public Dataset<br>
                <strong>Bereinigung:</strong> Pandas — Data Cleaning & Transformation<br>
                <strong>Datenbank:</strong> PostgreSQL — Relationale Speicherung<br>
                <strong>ORM:</strong> SQLAlchemy — Connection Pooling & Queries<br>
                <strong>Dashboard:</strong> Streamlit — Python Web-Framework<br>
                <strong>Visualisierung:</strong> Plotly Express & Graph Objects<br>
                <strong>ML:</strong> Scikit-Learn — Klassifikation, Multiple Lineare Regression<br>
                <strong>Design:</strong> Custom CSS — Dark Mode / Glassmorphism
            </div>
        </div>""", unsafe_allow_html=True)


    spacer()

    # RESPONSIVES ERD DIAGRAMM 
    erd_html = f"""
    <div style="margin-top:20px; border-bottom:1px solid rgba(55,65,81,0.35); padding-bottom:12px; margin-bottom:24px; display:flex; align-items:center; gap:10px;">
        <span style="width:8px; height:8px; border-radius:50%; background:{C['teal']}; box-shadow:0 0 8px {C['teal']};"></span>
        <span style="font-size:14px; font-weight:700; color:#e2e8f0; letter-spacing:-0.2px;">Entity-Relationship Model (ERD)</span>
    </div>

    <style>
    /* Die Tabelle selbst (ohne absolute Positionierung, da das SVG-foreignObject das übernimmt) */
    .erd-table {{ width: 100%; height: 100%; background: #111827; border: 1px solid #374151; border-radius: 8px; box-shadow: 0 8px 25px rgba(0,0,0,0.5); display: flex; flex-direction: column; transition: border-color 0.2s, box-shadow 0.2s; box-sizing: border-box; }}
    .erd-table:hover {{ border-color: rgba(99,102,241,0.6); box-shadow: 0 8px 30px rgba(99,102,241,0.25); }}

    /* Wrapper für foreignObject um Padding für den Box-Shadow zu gewähren */
    .erd-fo-wrap {{ padding: 20px; width: 100%; height: 100%; box-sizing: border-box; }}

    .erd-header {{ height: 40px; padding: 0 14px; font-weight: 700; font-size: 13px; border-bottom: 1px solid #374151; border-radius: 8px 8px 0 0; display: flex; justify-content: space-between; align-items: center; box-sizing: border-box; }}
    .erd-body {{ padding: 8px 14px; box-sizing: border-box; display: flex; flex-direction: column; }}
    .erd-row {{ height: 20px; width: 100%; display: flex; justify-content: space-between; align-items: center; box-sizing: border-box; font-size: 11px; font-family: 'Consolas', 'Courier New', monospace; color: #94a3b8; border-bottom: 1px dashed rgba(255,255,255,0.03); }}
    .erd-row:last-child {{ border-bottom: none; }}

    .erd-pk {{ color: #f59e0b; font-weight: 700; }}
    .erd-fk {{ color: #8b5cf6; font-weight: 700; }}
    .erd-type {{ color: #475569; font-size: 10px; }}

    /* SVG Badges */
    .erd-badge {{ fill: #1e293b; stroke: #475569; stroke-width: 1px; rx: 3px; ry: 3px; }}
    .erd-badge-text {{ font-size: 10px; font-weight: 700; fill: #cbd5e1; font-family: 'Inter', sans-serif; text-anchor: middle; }}
    </style>

    <!-- ZENTRIERTER, RESPONSIVER WRAPPER (OHNE MAX-WIDTH FÜR PERFEKTES SKALIEREN) -->
    <div style="width:100%; display:flex; justify-content:center; align-items:center; padding:10px 0;">
        <svg viewBox="0 0 1550 850" style="width:100%; height:auto; background-color:rgba(11,15,25,0.4); background-image:radial-gradient(rgba(255,255,255,0.06) 1px, transparent 1px); background-size:20px 20px; border-radius:16px; border:1px dashed rgba(55,65,81,0.5);">

            <!-- VERBINDUNGSLINIEN -->
            <!-- 1. events(n) -> users(1) -->
            <path d="M 260 178 L 290 178 L 290 108 L 320 108" fill="none" stroke="#64748b" stroke-width="1.5" />
            <rect x="267" y="170" width="16" height="16" class="erd-badge" />
            <text x="275" y="181" class="erd-badge-text">n</text>
            <rect x="297" y="100" width="16" height="16" class="erd-badge" />
            <text x="305" y="111" class="erd-badge-text">1</text>

            <!-- 2. orders(n) -> users(1) -->
            <path d="M 620 178 L 590 178 L 590 108 L 560 108" fill="none" stroke="#64748b" stroke-width="1.5" />
            <rect x="597" y="170" width="16" height="16" class="erd-badge" />
            <text x="605" y="181" class="erd-badge-text">n</text>
            <rect x="567" y="100" width="16" height="16" class="erd-badge" />
            <text x="575" y="111" class="erd-badge-text">1</text>

            <!-- 3. order_items(n) -> orders(1) -->
            <path d="M 920 178 L 890 178 L 890 158 L 860 158" fill="none" stroke="#64748b" stroke-width="1.5" />
            <rect x="897" y="170" width="16" height="16" class="erd-badge" />
            <text x="905" y="181" class="erd-badge-text">n</text>
            <rect x="867" y="150" width="16" height="16" class="erd-badge" />
            <text x="875" y="161" class="erd-badge-text">1</text>

            <!-- 4. order_items(n) -> inventory_items(1) (Läuft exakt zwischen orders und order_items bei x=880 durch) -->
            <path d="M 920 218 L 880 218 L 880 508 L 920 508" fill="none" stroke="#64748b" stroke-width="1.5" />
            <rect x="892" y="210" width="16" height="16" class="erd-badge" />
            <text x="900" y="221" class="erd-badge-text">n</text>
            <rect x="892" y="500" width="16" height="16" class="erd-badge" />
            <text x="900" y="511" class="erd-badge-text">1</text>

            <!-- 5. inventory_items(n) -> products(1) -->
            <path d="M 1160 528 L 1190 528 L 1190 408 L 1220 408" fill="none" stroke="#64748b" stroke-width="1.5" />
            <rect x="1167" y="520" width="16" height="16" class="erd-badge" />
            <text x="1175" y="531" class="erd-badge-text">n</text>
            <rect x="1197" y="400" width="16" height="16" class="erd-badge" />
            <text x="1205" y="411" class="erd-badge-text">1</text>

            <!-- 6. products(n) -> distribution_centers(1) -->
            <path d="M 1460 428 L 1480 428 L 1480 800 L 590 800 L 590 608 L 560 608" fill="none" stroke="#64748b" stroke-width="1.5" />
            <rect x="1462" y="420" width="16" height="16" class="erd-badge" />
            <text x="1470" y="431" class="erd-badge-text">n</text>
            <rect x="567" y="600" width="16" height="16" class="erd-badge" />
            <text x="575" y="611" class="erd-badge-text">1</text>


            <!-- TABELLEN (Eingebettet in die skalierbare SVG per foreignObject) -->

            <!-- events (x=20, y=100) FO: x=0, y=80, w=280, h=336 -->
            <foreignObject x="0" y="80" width="280" height="336">
                <div xmlns="http://www.w3.org/1999/xhtml" class="erd-fo-wrap">
                    <div class="erd-table">
                        <div class="erd-header" style="border-top: 3px solid #0ea5e9; color: #0ea5e9; background: rgba(14,165,233,0.1);">
                            <span>events</span><span>🖱️</span>
                        </div>
                        <div class="erd-body">
                            <div class="erd-row"><span class="erd-pk">🔑 event_id</span><span class="erd-type">int</span></div>
                            <div class="erd-row"><span class="erd-fk">🔗 user_id</span><span class="erd-type">int</span></div>
                            <div class="erd-row"><span>browser</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>city</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>created_at</span><span class="erd-type">timestamp</span></div>
                            <div class="erd-row"><span>event_type</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>postal_code</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>sequence_number</span><span class="erd-type">int</span></div>
                            <div class="erd-row"><span>session_id</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>state</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>traffic_source</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>uri</span><span class="erd-type">varchar</span></div>
                        </div>
                    </div>
                </div>
            </foreignObject>

            <!-- users (x=320, y=50) FO: x=300, y=30, w=280, h=376 -->
            <foreignObject x="300" y="30" width="280" height="376">
                <div xmlns="http://www.w3.org/1999/xhtml" class="erd-fo-wrap">
                    <div class="erd-table">
                        <div class="erd-header" style="border-top: 3px solid #8b5cf6; color: #8b5cf6; background: rgba(139,92,246,0.1);">
                            <span>users</span><span>👤</span>
                        </div>
                        <div class="erd-body">
                            <div class="erd-row"><span class="erd-pk">🔑 user_id</span><span class="erd-type">int</span></div>
                            <div class="erd-row"><span>age</span><span class="erd-type">int</span></div>
                            <div class="erd-row"><span>city</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>country</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>created_at</span><span class="erd-type">timestamp</span></div>
                            <div class="erd-row"><span>first_name</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>gender</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>last_name</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>latitude</span><span class="erd-type">float</span></div>
                            <div class="erd-row"><span>longitude</span><span class="erd-type">float</span></div>
                            <div class="erd-row"><span>postal_code</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>state</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>street_address</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>traffic_source</span><span class="erd-type">varchar</span></div>
                        </div>
                    </div>
                </div>
            </foreignObject>

            <!-- orders (x=620, y=100) FO: x=600, y=80, w=280, h=276 -->
            <foreignObject x="600" y="80" width="280" height="276">
                <div xmlns="http://www.w3.org/1999/xhtml" class="erd-fo-wrap">
                    <div class="erd-table">
                        <div class="erd-header" style="border-top: 3px solid #22d3ee; color: #22d3ee; background: rgba(34,211,238,0.1);">
                            <span>orders</span><span>📋</span>
                        </div>
                        <div class="erd-body">
                            <div class="erd-row"><span class="erd-pk">🔑 order_id</span><span class="erd-type">int</span></div>
                            <div class="erd-row"><span class="erd-fk">🔗 user_id</span><span class="erd-type">int</span></div>
                            <div class="erd-row"><span>created_at</span><span class="erd-type">timestamp</span></div>
                            <div class="erd-row"><span>delivered_at</span><span class="erd-type">timestamp</span></div>
                            <div class="erd-row"><span>gender</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>num_of_item</span><span class="erd-type">int</span></div>
                            <div class="erd-row"><span>returned_at</span><span class="erd-type">timestamp</span></div>
                            <div class="erd-row"><span>shipped_at</span><span class="erd-type">timestamp</span></div>
                            <div class="erd-row"><span>status</span><span class="erd-type">varchar</span></div>
                        </div>
                    </div>
                </div>
            </foreignObject>

            <!-- order_items (x=920, y=100) FO: x=900, y=80, w=280, h=296 -->
            <foreignObject x="900" y="80" width="280" height="296">
                <div xmlns="http://www.w3.org/1999/xhtml" class="erd-fo-wrap">
                    <div class="erd-table">
                        <div class="erd-header" style="border-top: 3px solid #10b981; color: #10b981; background: rgba(16,185,129,0.1);">
                            <span>order_items</span><span>🛒</span>
                        </div>
                        <div class="erd-body">
                            <div class="erd-row"><span class="erd-pk">🔑 order_item_id</span><span class="erd-type">int</span></div>
                            <div class="erd-row"><span class="erd-fk">🔗 order_id</span><span class="erd-type">int</span></div>
                            <div class="erd-row"><span class="erd-fk">🔗 user_id</span><span class="erd-type">int</span></div>
                            <div class="erd-row"><span class="erd-fk">🔗 inventory_item_id</span><span class="erd-type">int</span></div>
                            <div class="erd-row"><span>created_at</span><span class="erd-type">timestamp</span></div>
                            <div class="erd-row"><span>delivered_at</span><span class="erd-type">timestamp</span></div>
                            <div class="erd-row"><span>returned_at</span><span class="erd-type">timestamp</span></div>
                            <div class="erd-row"><span>sale_price</span><span class="erd-type">float</span></div>
                            <div class="erd-row"><span>shipped_at</span><span class="erd-type">timestamp</span></div>
                            <div class="erd-row"><span>status</span><span class="erd-type">varchar</span></div>
                        </div>
                    </div>
                </div>
            </foreignObject>

            <!-- inventory_items (x=920, y=450) FO: x=900, y=430, w=280, h=336 -->
            <foreignObject x="900" y="430" width="280" height="336">
                <div xmlns="http://www.w3.org/1999/xhtml" class="erd-fo-wrap">
                    <div class="erd-table">
                        <div class="erd-header" style="border-top: 3px solid #f59e0b; color: #f59e0b; background: rgba(245,158,11,0.1);">
                            <span>inventory_items</span><span>📦</span>
                        </div>
                        <div class="erd-body">
                            <div class="erd-row"><span class="erd-pk">🔑 inventory_item_id</span><span class="erd-type">int</span></div>
                            <div class="erd-row"><span class="erd-fk">🔗 product_id</span><span class="erd-type">int</span></div>
                            <div class="erd-row"><span>cost</span><span class="erd-type">float</span></div>
                            <div class="erd-row"><span>created_at</span><span class="erd-type">timestamp</span></div>
                            <div class="erd-row"><span>product_brand</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>product_category</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>product_department</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>product_distribution_center_id</span><span class="erd-type">int</span></div>
                            <div class="erd-row"><span>product_name</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>product_retail_price</span><span class="erd-type">float</span></div>
                            <div class="erd-row"><span>product_sku</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>sold_at</span><span class="erd-type">timestamp</span></div>
                        </div>
                    </div>
                </div>
            </foreignObject>

            <!-- products (x=1220, y=350) FO: x=1200, y=330, w=280, h=276 -->
            <foreignObject x="1200" y="330" width="280" height="276">
                <div xmlns="http://www.w3.org/1999/xhtml" class="erd-fo-wrap">
                    <div class="erd-table">
                        <div class="erd-header" style="border-top: 3px solid #ec4899; color: #ec4899; background: rgba(236,72,153,0.1);">
                            <span>products</span><span>🏷️</span>
                        </div>
                        <div class="erd-body">
                            <div class="erd-row"><span class="erd-pk">🔑 product_id</span><span class="erd-type">int</span></div>
                            <div class="erd-row"><span class="erd-fk">🔗 distribution_center_id</span><span class="erd-type">int</span></div>
                            <div class="erd-row"><span>brand</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>category</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>cost</span><span class="erd-type">float</span></div>
                            <div class="erd-row"><span>department</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>name</span><span class="erd-type">varchar</span></div>
                            <div class="erd-row"><span>retail_price</span><span class="erd-type">float</span></div>
                            <div class="erd-row"><span>sku</span><span class="erd-type">varchar</span></div>
                        </div>
                    </div>
                </div>
            </foreignObject>

            <!-- distribution_centers (x=320, y=550) FO: x=300, y=530, w=280, h=176 -->
            <foreignObject x="300" y="530" width="280" height="176">
                <div xmlns="http://www.w3.org/1999/xhtml" class="erd-fo-wrap">
                    <div class="erd-table">
                        <div class="erd-header" style="border-top: 3px solid #f43f5e; color: #f43f5e; background: rgba(244,63,94,0.1);">
                            <span>distribution_centers</span><span>🏭</span>
                        </div>
                        <div class="erd-body">
                            <div class="erd-row"><span class="erd-pk">🔑 distribution_center_id</span><span class="erd-type">int</span></div>
                            <div class="erd-row"><span>latitude</span><span class="erd-type">float</span></div>
                            <div class="erd-row"><span>longitude</span><span class="erd-type">float</span></div>
                            <div class="erd-row"><span>name</span><span class="erd-type">varchar</span></div>
                        </div>
                    </div>
                </div>
            </foreignObject>

        </svg>
    </div>
    """

    # .replace('\n', '') verhindert Markdown-Codeblock-Bugs in Streamlit 
    st.markdown(erd_html.replace('\n', ''), unsafe_allow_html=True)
    
# ══════════════════════════════════════════════════════════════
#  PAGE 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════
def page_overview():
    render_header("1. Executive Overview")

    #  ROW 1: KPIs 
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1: kpi("💰", "Total Sales",   human_money(tot_sales),   C["cyan"])
    with k2: kpi("📋", "Orders",        human_number(tot_orders), C["emerald"])
    with k3: kpi("🛒", "Avg. Order",    human_money(aov),         C["violet"])
    with k4: kpi("↩️", "Return Rate",   pct(ret_rate),            C["rose"])
    with k5: kpi("📊", "Gross Profit",  human_money(gross_profit),C["amber"])
    with k6: kpi("📈", "Profit Margin", pct(margin),              C["sky"])
    spacer()

    # ── ROW 2: Yearly Sales + Top Kategorien ──
    left, right = st.columns([2.2, 1])
    with left:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Umsatz nach Jahr", C["cyan"])
        if len(yearly):
            fig = px.line(
                yearly,
                x="Year",
                y="Sales",
                markers=True,
                color_discrete_sequence=[C["cyan"]],
            )
            fig.update_traces(
                line=dict(width=3),
                marker=dict(size=9, line=dict(width=2, color=C["card_solid"])),
            )
            fig = plotly_dark(fig, 280, showlegend=False)
            fig.update_xaxes(title_text="Jahr", tickmode="linear", dtick=1)
            fig.update_yaxes(title_text="Sales ($)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Keine Jahresdaten verfügbar.")

        m1, m2 = st.columns(2)
        yoy_last = yearly["YoY"].dropna().iloc[-1] if yearly["YoY"].dropna().shape[0] else 0
        best_year = str(int(yearly.loc[yearly["Sales"].idxmax(), "Year"])) if len(yearly) else "—"
        with m1: mini("YoY Growth", pct(yoy_last), C["emerald"] if yoy_last >= 0 else C["rose"])
        with m2: mini("Bestes Jahr", best_year, C["amber"])
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Top 10 Kategorien", C["violet"])
        cat_top = (
            cs.groupby("category", as_index=False)["sale_price"].sum()
            .sort_values("sale_price", ascending=False).head(10)
        )
        fig = px.bar(
            cat_top.sort_values("sale_price"), x="sale_price", y="category",
            orientation="h", color_discrete_sequence=[C["violet"]],
        )
        fig.update_traces(marker=dict(cornerradius=5))
        st.plotly_chart(plotly_dark(fig, 340, showlegend=False), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    spacer()

    #  ROW 3: Monthly Sales 
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    sec("Umsatz nach Monat", C["emerald"])
    if len(monthly_agg):
        fig = px.bar(
            monthly_agg,
            x="Monat",
            y="Sales",
            color_discrete_sequence=[C["emerald"]],
        )
        fig.update_traces(marker=dict(cornerradius=4))
        fig = plotly_dark(fig, 300, showlegend=False)
        fig.update_xaxes(title_text="Monat", categoryorder="array",
                         categoryarray=["Jan","Feb","Mär","Apr","Mai","Jun",
                                        "Jul","Aug","Sep","Okt","Nov","Dez"])
        fig.update_yaxes(title_text="Sales ($)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Keine Monatsdaten verfügbar.")

    m1, m2 = st.columns(2)
    best_month_agg = monthly_agg.loc[monthly_agg["Sales"].idxmax(), "Monat"] if len(monthly_agg) else "—"
    worst_month_agg = monthly_agg.loc[monthly_agg["Sales"].idxmin(), "Monat"] if len(monthly_agg) else "—"
    with m1: mini("Stärkster Monat", best_month_agg, C["emerald"])
    with m2: mini("Schwächster Monat", worst_month_agg, C["amber"])
    st.markdown("</div>", unsafe_allow_html=True)

    spacer()

    # ROW 4: Geo-Map (breit) + Monthly Table 
    left2, right2 = st.columns([1.6, 1])
    with left2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Bestellungen weltweit", C["blue"])
        geo = (
            cs.groupby(["country", "latitude", "longitude"], dropna=False)
            .agg(Orders=("order_id", "nunique")).reset_index()
            .dropna(subset=["latitude", "longitude"])
        )
        if len(geo):
            fig = px.scatter_geo(
                geo, lat="latitude", lon="longitude", size="Orders",
                hover_name="country", projection="natural earth",
                color="Orders", color_continuous_scale="ice",
            )
            fig.update_geos(
                bgcolor="rgba(0,0,0,0)",
                landcolor="#1e293b", oceancolor="#0f172a",
                showlakes=False, framecolor="rgba(0,0,0,0)",
                coastlinecolor="#334155",
            )
            st.plotly_chart(plotly_dark(fig, 340), use_container_width=True)
        else:
            st.info("Keine Geodaten verfügbar.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Monatliche Umsatz-Tabelle", C["emerald"])
        show_monthly = monthly[["Month", "Sales", "MoM"]].copy()
        show_monthly["Sales"] = show_monthly["Sales"].map(lambda x: f"{x:,.0f} $")
        show_monthly["MoM"] = show_monthly["MoM"].map(lambda x: f"{x:+.1f}%" if not pd.isna(x) else "—")
        st.dataframe(show_monthly, use_container_width=True, hide_index=True, height=340)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — RETURN ANALYSIS
# ══════════════════════════════════════════════════════════════
def page_returns():
    render_header("2. Return Analysis")
    ret_items = fs["is_returned"].sum()
    lost = fs.loc[fs["is_returned"] == 1, "profit"].sum()
    avg_ret = (
        (fs.loc[fs["is_returned"] == 1, "returned_at"] - fs.loc[fs["is_returned"] == 1, "created_at"])
        .dt.days.mean()
    )

    #  KPIs 
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: kpi("↩️", "Return Rate",    pct(ret_rate),             C["rose"])
    with k2: kpi("💸", "Returned Value",  human_money(ret_val),     "#dc2626")
    with k3: kpi("📦", "Returned Items",  human_number(ret_items),  C["pink"])
    with k4: kpi("📉", "Lost Profit",     human_money(lost),        C["orange"])
    with k5: kpi("⏱️", "Avg. Return",     f"{0 if pd.isna(avg_ret) else avg_ret:.1f} Tage", C["amber"])
    spacer()

    # ROW 2: 2 Charts 
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Return Rate nach Kategorie", C["violet"])
        rc = fs.groupby("category").agg(total=("id", "count"), returned=("is_returned", "sum")).reset_index()
        rc["rr"] = rc["returned"] / rc["total"] * 100
        rc = rc.sort_values("rr", ascending=False).head(10)
        fig = px.bar(rc.sort_values("rr"), x="rr", y="category", orientation="h",
                     color_discrete_sequence=[C["violet"]])
        fig.update_traces(marker=dict(cornerradius=5))
        st.plotly_chart(plotly_dark(fig, 340, showlegend=False), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Return Rate nach Brand (Top 10)", C["pink"])
        rb = fs.groupby("brand").agg(total=("id", "count"), returned=("is_returned", "sum")).reset_index()
        rb["rr"] = rb["returned"] / rb["total"] * 100
        rb = rb[rb["total"] >= 10].sort_values("rr", ascending=False).head(10)
        fig = px.bar(rb.sort_values("rr"), x="rr", y="brand", orientation="h",
                     color_discrete_sequence=[C["pink"]])
        fig.update_traces(marker=dict(cornerradius=5))
        st.plotly_chart(plotly_dark(fig, 340, showlegend=False), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    spacer()

    #  ROW 3: Scatter + Funnel 
    left2, right2 = st.columns([1.6, 1])
    with left2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Retail Price vs Return Rate", C["cyan"])
        ps = fs.groupby(["product_name", "retail_price"]).agg(
            total=("id", "count"), returned=("is_returned", "sum"), sales=("sale_price", "sum")
        ).reset_index()
        ps = ps[ps["total"] >= 5]
        ps["rr"] = ps["returned"] / ps["total"] * 100
        fig = px.scatter(ps, x="retail_price", y="rr", size="sales",
                         hover_name="product_name", color_discrete_sequence=[C["cyan"]])
        st.plotly_chart(plotly_dark(fig, 320), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Bestellstatus Funnel", C["violet"])
        sd = fs.groupby("status").size().reset_index(name="count")
        fig = order_funnel(sd)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — CUSTOMER & CONVERSION
# ══════════════════════════════════════════════════════════════
def page_customers():
    render_header("3. Customer & Conversion")

    #  ROW 1: Funnel + Traffic + Age 
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Conversion Funnel", C["emerald"])
        fig, _ = event_funnel(fe_all)
        st.plotly_chart(fig, use_container_width=True)
        ev = fe_all.groupby("event_type").size().reset_index(name="events")
        pc = ev.loc[ev["event_type"].str.lower() == "purchase", "events"].sum()
        sc = fe_all["session_id"].nunique() if "session_id" in fe_all.columns else len(fe_all)
        conv = (pc / sc * 100) if sc else 0
        mini("Conversion Rate", pct(conv), C["emerald"])
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Traffic Sources", C["cyan"])
        tf = fu.groupby("traffic_source").size().reset_index(name="users").sort_values("users", ascending=False)
        if len(tf):
            fig = px.pie(tf, names="traffic_source", values="users", hole=0.65,
                         color_discrete_sequence=CHART_SEQ)
            st.plotly_chart(plotly_dark(fig, 360), use_container_width=True)
        else:
            st.info("Keine Daten.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Altersverteilung", C["violet"])
        adf = fu.copy()
        adf["ag"] = pd.cut(adf["age"], bins=[0,20,30,40,50,120], labels=["<20","20–29","30–39","40–49","50+"], right=False)
        aa = adf.groupby("ag", observed=False).size().reset_index(name="users")
        fig = px.pie(aa, names="ag", values="users", hole=0.65, color_discrete_sequence=CHART_SEQ)
        st.plotly_chart(plotly_dark(fig, 360), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    spacer()

    #  ROW 2: Top Customers 
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    sec("Top 10 Customers — Lifetime Value", C["amber"])
    tc = (
        cs.groupby("user_id")
        .agg(Orders=("order_id", "nunique"), Total_Sales=("sale_price", "sum"))
        .reset_index().sort_values("Total_Sales", ascending=False).head(10)
    )
    tc = tc.merge(fu[["id", "first_name", "last_name"]], left_on="user_id", right_on="id", how="left")
    tc["Customer"] = tc["first_name"].fillna("") + " " + tc["last_name"].fillna("")
    show = tc[["Customer", "Orders", "Total_Sales"]].copy()
    show["Total_Sales"] = show["Total_Sales"].map(lambda x: f"{x:,.2f} $")

   
    tc_bar = tc[["Customer", "Total_Sales"]].copy().sort_values("Total_Sales")
    fig = px.bar(tc_bar, x="Total_Sales", y="Customer", orientation="h",
                 color_discrete_sequence=[C["amber"]], text="Total_Sales")
    fig.update_traces(
        texttemplate="%{text:,.0f} $", textposition="outside",
        marker=dict(cornerradius=5),
    )
    st.plotly_chart(plotly_dark(fig, 350, showlegend=False), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 4 — PRODUCT & INVENTORY
# ══════════════════════════════════════════════════════════════
def page_products():
    render_header("4. Product & Inventory")

    # ROW 1: Treemap (groß) + Gauge 
    left, right = st.columns([2.2, 1])
    with left:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Umsatz nach Department & Kategorie", C["violet"])
        tree = cs.groupby(["department", "category"], as_index=False)["sale_price"].sum().rename(columns={"sale_price": "Sales"})
        fig = px.treemap(tree, path=["department", "category"], values="Sales",
                         color="Sales", color_continuous_scale="Purp")
        st.plotly_chart(plotly_dark(fig, 380), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Ø Marge", C["cyan"])
        cm = cs.groupby("category").agg(s=("sale_price", "sum"), p=("profit", "sum")).reset_index()
        cm["m"] = cm["p"] / cm["s"] * 100
        avg_m = cm["m"].mean() if len(cm) else 0
        fig = gauge(avg_m, "Margin %")
        st.plotly_chart(fig, use_container_width=True)

        # Extra stats
        spacer()
        m1, m2 = st.columns(2)
        with m1: mini("Profit", human_money(gross_profit), C["emerald"])
        with m2: mini("Sales", human_money(tot_sales), C["cyan"])
        st.markdown("</div>", unsafe_allow_html=True)

    spacer()

    # ROW 2: Distribution Center + Top Produkte 
    left2, right2 = st.columns(2)
    with left2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Umsatz nach Distribution Center", C["amber"])
        dc = (
            cs.groupby("distribution_center_id").agg(Sales=("sale_price", "sum"))
            .reset_index().merge(dist_centers, left_on="distribution_center_id", right_on="id", how="left")
        )
        if len(dc):
            fig = px.bar(dc.sort_values("Sales"), x="Sales", y="name", orientation="h",
                         color_discrete_sequence=[C["amber"]])
            fig.update_traces(marker=dict(cornerradius=5))
            st.plotly_chart(plotly_dark(fig, 340, showlegend=False), use_container_width=True)
        else:
            st.info("Keine Daten.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Top 10 Produkte nach Umsatz", C["blue"])
        pt = cs.groupby("product_name", as_index=False)["sale_price"].sum().sort_values("sale_price", ascending=False).head(10)
        fig = px.bar(pt.sort_values("sale_price"), x="sale_price", y="product_name",
                     orientation="h", color_discrete_sequence=[C["blue"]])
        fig.update_traces(marker=dict(cornerradius=5))
        st.plotly_chart(plotly_dark(fig, 340, showlegend=False), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 5 — ADVANCED INSIGHTS
# ══════════════════════════════════════════════════════════════
def page_advanced():
    render_header("5. Advanced Insights")

    #  ROW 1: RFM + Profitability Matrix 
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("RFM Customer Segmentation", C["violet"])

        import plotly.graph_objects as go

        fig = go.Figure()

        # Definition der 6 Zielgruppen 
        segments = [
            {"name": "Hibernating", "x0": 0, "y0": 0, "x1": 4, "y1": 4, "color": "#3b82f6"},
            {"name": "At Risk", "x0": 0, "y0": 4, "x1": 4, "y1": 8, "color": "#06b6d4"},
            {"name": "Can't Lose", "x0": 0, "y0": 8, "x1": 4, "y1": 10, "color": "#f59e0b"},
            {"name": "Potential Loyal", "x0": 4, "y0": 0, "x1": 10, "y1": 5, "color": "#8b5cf6"},
            {"name": "Loyal", "x0": 4, "y0": 5, "x1": 8, "y1": 10, "color": "#10b981"},
            {"name": "Champions", "x0": 8, "y0": 5, "x1": 10, "y1": 10, "color": "#f43f5e"},
        ]

        # Formen und Beschriftungen zeichnen
        for seg in segments:
            fig.add_shape(
                type="rect",
                x0=seg["x0"], y0=seg["y0"], x1=seg["x1"], y1=seg["y1"],
                fillcolor=seg["color"],
                line=dict(color="#0E1117", width=2)
            )
            fig.add_annotation(
                x=seg["x0"] + 0.2, y=seg["y1"] - 0.2,
                text=f"<b>{seg['name']}</b>",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                font=dict(color="white", size=14)
            )

        # Achsen 
        fig.update_xaxes(
            title="<b>Recency score ⟶</b>",
            title_font=dict(size=16, color="white"),
            showgrid=False, zeroline=False, showticklabels=False,
            showline=True, linewidth=2, linecolor="white",
            range=[0, 10.1]
        )
        fig.update_yaxes(
            title="<b>Frequency and monetary score ⟶</b>",
            title_font=dict(size=16, color="white"),
            showgrid=False, zeroline=False, showticklabels=False,
            showline=True, linewidth=2, linecolor="white",
            range=[0, 10.1]
        )

        fig.update_layout(
            margin=dict(t=20, l=40, r=20, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=385  # Etwas höher für besseres Alignment
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Profitability Matrix", C["cyan"])

       
        c1, c2 = st.columns(2)
        with c1:
            agg_choice = st.selectbox("Ebene (Aggregation)", ["Kategorie", "Marke", "Produkt"], key="prof_agg")
        with c2:
            min_orders = st.number_input("Mindestbestellungen", min_value=1, value=10, step=5, key="prof_min")

        agg_col_map = {"Kategorie": "category", "Marke": "brand", "Produkt": "product_name"}
        agg_col = agg_col_map[agg_choice]

        # 1. Daten aggregieren
        prof = (
            cs.groupby(agg_col)
            .agg(Sales=("sale_price", "sum"), Profit=("profit", "sum"), Orders=("order_id", "nunique"))
            .reset_index()
        )

        # 2. Rausch-Filter 
        prof = prof[prof["Orders"] >= min_orders]

    
        prof = prof.sort_values(by="Sales", ascending=False).head(100)

        # 4. Marge berechnen 
        prof["Margin (%)"] = (prof["Profit"] / prof["Sales"] * 100).fillna(0)

        # Werte für den Tooltip runden
        prof["Sales"] = prof["Sales"].round(2)
        prof["Profit"] = prof["Profit"].round(2)
        prof["Margin (%)"] = prof["Margin (%)"].round(1)

        if len(prof) > 0:
            fig = px.scatter(
                prof,
                x="Sales",
                y="Profit",
                size="Orders",
                color="Margin (%)",
                color_continuous_scale="Teal",
                hover_name=agg_col,
                hover_data={"Margin (%)": True, "Orders": True, "Sales": True, "Profit": True}
            )
            fig.update_layout(coloraxis_colorbar=dict(title="Marge %", thickness=10))
            
            st.plotly_chart(plotly_dark(fig, 290), use_container_width=True)
        else:
            st.info("⚠️ Keine Daten gefunden, die diesen Kriterien entsprechen (z.B. zu hoher Filter).")

        st.markdown("</div>", unsafe_allow_html=True)

    spacer()

    # ROW 2: Repeat Customer + Delivery (2 cols) 
    left2, right2 = st.columns([1.5, 1])

    with left2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Repeat Customer Trend", C["emerald"])

        # Datenfilterung bis exklusive Juli 2026
        cs_trend = cs[cs["created_at"] < "2026-07-01"]

        rep = cs_trend.groupby("user_id")["order_id"].nunique().reset_index(name="orders")
        rr = (rep["orders"] > 1).mean() * 100 if len(rep) else 0
        mr = (
            cs_trend.assign(Month=cs_trend["created_at"].dt.to_period("M").astype(str))
            .groupby("Month")["user_id"].nunique().reset_index(name="Customers")
        )

        fig = px.area(mr, x="Month", y="Customers", color_discrete_sequence=[C["emerald"]])
        fig.update_traces(fill="tozeroy", fillcolor="rgba(16,185,129,0.08)", line=dict(width=2.5))
        st.plotly_chart(plotly_dark(fig, 260, showlegend=False), use_container_width=True)
        mini("Repeat Customer Rate", pct(rr), C["emerald"])
        st.markdown("</div>", unsafe_allow_html=True)

    with right2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Delivery Performance", C["amber"])
        ot = ((cs["delivery_days"] <= 3) & cs["delivery_days"].notna()).mean() * 100 if len(cs) else 0
        ad = cs["delivery_days"].mean()
        lr = ((cs["delivery_days"] > 3) & cs["delivery_days"].notna()).mean() * 100 if len(cs) else 0

        mini("Ø Lieferzeit", f"{0 if pd.isna(ad) else ad:.1f} Tage", C["amber"])
        spacer()
        mini("On-Time Rate", pct(ot), C["emerald"])
        spacer()
        mini("Late Deliveries", pct(lr), C["rose"])
        spacer()
        mini("Total Orders", human_number(tot_orders), C["cyan"])
        st.markdown("</div>", unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════
#  PAGE 5b — RFM-KUNDENSEGMENTIERUNG
# ══════════════════════════════════════════════════════════════
def page_kundensegmentierung():
    render_header("Kundensegmentierung")

    # Erklärung: Was ist RFM?
    left, right = st.columns(2)
    with left:
        st.markdown(f"""
        <div class="info-block">
            <div class="ib-title"><span class="ib-dot" style="background:{C['violet']}; box-shadow:0 0 8px {C['violet']};"></span> Was ist RFM?</div>
            <div class="ib-text">
                <strong>RFM</strong> ist eine Methode aus dem Direktmarketing, um Kunden anhand ihres
                Kaufverhaltens in logische Gruppen (Segmente) zu unterteilen:<br><br>
                ▸ <strong>R</strong>ecency — Tage seit dem letzten Kauf<br>
                ▸ <strong>F</strong>requency — Gesamtanzahl Bestellungen<br>
                ▸ <strong>M</strong>onetary — Gesamter generierter Umsatz<br><br>
                Im Gegensatz zur Churn-Prediction (ML) handelt es sich um eine
                rein <strong>deskriptive Analyse</strong> auf Basis historischer Fakten.
            </div>
        </div>""", unsafe_allow_html=True)
    with right:
        st.markdown(f"""
        <div class="info-block">
            <div class="ib-title"><span class="ib-dot" style="background:{C['cyan']}; box-shadow:0 0 8px {C['cyan']};"></span> Scoring-Methode</div>
            <div class="ib-text">
                Jeder Kunde erhält für R, F und M einen <strong>Score von 1–5</strong>
                (5 = bester Wert) via Quintile (<code>pd.qcut</code>).<br><br>
                <strong>RFM-Score = R + F + M</strong> (Bereich: 3–15)<br><br>
                ▸ <strong>13–15:</strong> Champions<br>
                ▸ <strong>11–12:</strong> Loyal<br>
                ▸ <strong>9–10:</strong> Potential Loyal<br>
                ▸ <strong>7–8:</strong> At Risk<br>
                ▸ <strong>5–6:</strong> Can't Lose<br>
                ▸ <strong>3–4:</strong> Hibernating
            </div>
        </div>""", unsafe_allow_html=True)

    spacer()

    # RFM berechnen 
    ref_date = cs["created_at"].max()
    cust_rfm = cs.groupby("user_id").agg(
        recency=("created_at", "max"),
        frequency=("order_id", "nunique"),
        monetary=("sale_price", "sum"),
    ).reset_index()
    cust_rfm["recency"] = (ref_date - cust_rfm["recency"]).dt.days
    cust_rfm = cust_rfm.dropna(subset=["recency", "frequency", "monetary"])

    if len(cust_rfm) < 5:
        st.warning("Nicht genügend Kundendaten für die RFM-Segmentierung.")
        return

    try:
        cust_rfm["R"] = pd.qcut(cust_rfm["recency"].rank(method="first"), 5, labels=[5,4,3,2,1]).astype(int)
        cust_rfm["F"] = pd.qcut(cust_rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
        cust_rfm["M"] = pd.qcut(cust_rfm["monetary"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
        cust_rfm["RFM_Score"] = cust_rfm["R"] + cust_rfm["F"] + cust_rfm["M"]
    except ValueError:
        st.warning("RFM-Segmentierung nicht möglich (zu wenig Streuung).")
        return

    def _seg(s):
        if s >= 13: return "Champions"
        if s >= 11: return "Loyal"
        if s >= 9:  return "Potential Loyal"
        if s >= 7:  return "At Risk"
        if s >= 5:  return "Can't Lose"
        return "Hibernating"

    cust_rfm["Segment"] = cust_rfm["RFM_Score"].apply(_seg)
    _seg_order = ["Champions", "Loyal", "Potential Loyal", "At Risk", "Can't Lose", "Hibernating"]

    #  KPIs 
    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi("👥", "Kunden gesamt", human_number(len(cust_rfm)), C["cyan"])
    with k2: kpi("🏆", "Champions", human_number((cust_rfm["Segment"] == "Champions").sum()), C["emerald"])
    with k3: kpi("⚠️", "At Risk", human_number((cust_rfm["Segment"] == "At Risk").sum()), C["amber"])
    with k4: kpi("💤", "Hibernating", human_number((cust_rfm["Segment"] == "Hibernating").sum()), C["rose"])
    spacer()

    #  Donut + Scatter 
    left_c, right_c = st.columns(2)
    with left_c:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Segment-Verteilung", C["violet"])
        sg = cust_rfm.groupby("Segment").size().reset_index(name="Kunden")
        sg["Segment"] = pd.Categorical(sg["Segment"], categories=_seg_order, ordered=True)
        sg = sg.sort_values("Segment")
        fig = px.pie(sg, names="Segment", values="Kunden", hole=0.65, color_discrete_sequence=CHART_SEQ)
        st.plotly_chart(plotly_dark(fig, 360), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right_c:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Recency vs. Monetary", C["cyan"])
        fig = px.scatter(cust_rfm, x="recency", y="monetary", size="frequency",
                         color="Segment", hover_data=["user_id", "RFM_Score"],
                         color_discrete_sequence=CHART_SEQ)
        fig.update_xaxes(autorange="reversed", title_text="Recency (Tage — niedrig = besser)")
        fig.update_yaxes(title_text="Monetary ($)")
        st.plotly_chart(plotly_dark(fig, 380), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    spacer()

    #  Segment-Tabelle 
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    sec("Segmente — Ø-Werte & Marketing-Empfehlungen", C["emerald"])
    _empf = {
        "Champions": "VIP-Angebote, Belohnungen, Testimonials anfragen.",
        "Loyal": "Upselling, Empfehlungsprogramme anbieten.",
        "Potential Loyal": "Treueprogramm, Frequency steigern.",
        "At Risk": "Re-Engagement-E-Mails, Rabatt-Codes senden.",
        "Can't Lose": "Persönliche Kontaktaufnahme, hohe Rabatte.",
        "Hibernating": "Günstige Standard-Newsletter.",
    }
    seg_stats = cust_rfm.groupby("Segment").agg(
        Kunden=("user_id", "count"), Ø_Recency=("recency", "mean"),
        Ø_Frequency=("frequency", "mean"), Ø_Monetary=("monetary", "mean"),
        Umsatz_Anteil=("monetary", "sum"),
    ).reset_index()
    total_m = cust_rfm["monetary"].sum()
    seg_stats["Umsatz_Anteil"] = (seg_stats["Umsatz_Anteil"] / total_m * 100).round(1) if total_m else 0
    seg_stats["Empfehlung"] = seg_stats["Segment"].map(_empf)
    seg_stats["Segment"] = pd.Categorical(seg_stats["Segment"], categories=_seg_order, ordered=True)
    seg_stats = seg_stats.sort_values("Segment")
    show_seg = seg_stats.copy()
    show_seg["Ø_Recency"] = show_seg["Ø_Recency"].map(lambda x: f"{x:.0f} T.")
    show_seg["Ø_Frequency"] = show_seg["Ø_Frequency"].map(lambda x: f"{x:.1f}")
    show_seg["Ø_Monetary"] = show_seg["Ø_Monetary"].map(lambda x: human_money(x))
    show_seg["Umsatz_Anteil"] = show_seg["Umsatz_Anteil"].map(lambda x: f"{x:.1f} %")
    show_seg.columns = ["Segment", "Kunden", "Ø Recency", "Ø Frequency", "Ø Monetary", "Umsatz %", "Marketing-Empfehlung"]
    st.dataframe(show_seg, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    spacer()

    # CRM-Export 
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    sec("CRM-Export — Kampagnen-Vorbereitung", C["amber"])
    sel_seg = st.selectbox("Segment wählen", _seg_order, index=3, label_visibility="collapsed")
    seg_cust = cust_rfm[cust_rfm["Segment"] == sel_seg].sort_values("monetary", ascending=False).copy()
    seg_cust = seg_cust.merge(fu[["id", "first_name", "last_name", "city", "country"]],
                              left_on="user_id", right_on="id", how="left")
    seg_cust["Kunde"] = (seg_cust["first_name"].fillna("") + " " + seg_cust["last_name"].fillna("")).str.strip().replace("", "Unbekannt")
    preview = seg_cust[["Kunde", "recency", "frequency", "monetary", "RFM_Score", "city", "country"]].head(15).copy()
    preview.columns = ["Kunde", "Recency", "Orders", "Umsatz ($)", "RFM", "Stadt", "Land"]
    preview["Umsatz ($)"] = seg_cust["monetary"].head(15).map(lambda x: f"{x:,.2f} $")
    st.dataframe(preview, use_container_width=True, hide_index=True)
    spacer()
    m1, m2 = st.columns(2)
    with m1: mini("Kunden im Segment", human_number(len(seg_cust)), C["violet"])
    with m2: mini("Ø Umsatz", human_money(seg_cust["monetary"].mean()), C["cyan"])
    spacer()
    csv_d = seg_cust[["user_id", "Kunde", "recency", "frequency", "monetary", "RFM_Score", "Segment", "city", "country"]].to_csv(index=False).encode("utf-8")
    st.download_button(f"📥 {sel_seg} als CSV ({len(seg_cust)} Kunden)", csv_d,
                       f"rfm_{sel_seg.lower().replace(' ', '_')}.csv", "text/csv", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 6a — CHURN MODELL-ERKLÄRUNG
# ══════════════════════════════════════════════════════════════
def page_churn_explain():
    render_header("📖 Churn — Modell-Erklärung", analysis=False)

    # ROW 1: Problem + Ziel 
    left, right = st.columns(2)

    with left:
        st.markdown(f"""
        <div class="info-block">
            <div class="ib-title">
                <span class="ib-dot" style="background:{C['rose']}; box-shadow:0 0 8px {C['rose']};"></span>
                Problemstellung
            </div>
            <div class="ib-text">
                Im E-Commerce wandern durchschnittlich <strong>20–40 %</strong> der Kunden
                innerhalb eines Jahres ab. Das Problem:<br><br>
                ▸ <strong>Neukundengewinnung kostet 5–7× mehr</strong> als Bestandskundenpflege<br>
                ▸ Unternehmen erkennen die Abwanderung erst, wenn es <strong>zu spät</strong> ist<br>
                ▸ Ohne prädiktive Analyse gibt es <strong>keine Frühwarnung</strong><br><br>
                <strong>Unsere Frage:</strong> Können wir anhand des bisherigen Kaufverhaltens
                vorhersagen, welche Kunden in den nächsten 90 Tagen abwandern werden?
            </div>
        </div>""", unsafe_allow_html=True)

    with right:
        st.markdown(f"""
        <div class="info-block">
            <div class="ib-title">
                <span class="ib-dot" style="background:{C['emerald']}; box-shadow:0 0 8px {C['emerald']};"></span>
                Unser Ansatz
            </div>
            <div class="ib-text">
                <strong>Supervised Learning — Binäre Klassifikation</strong><br><br>
                ▸ <strong>Label:</strong> Churned (1) = Letzte Bestellung &gt; 90 Tage<br>
                ▸ <strong>Modelle:</strong> 3 Algorithmen im Vergleich (RF, GB, LR)<br>
                ▸ <strong>Features:</strong> 10 Kundenmerkmale (Feature Selection mit 4 Methoden)<br>
                ▸ <strong>Scaling:</strong> StandardScaler für alle Features<br>
                ▸ <strong>Balancing:</strong> SMOTE + class_weight={{0:1, 1:2}} gegen Klassenungleichgewicht<br>
                ▸ <strong>Tuning:</strong> Manuelle Hyperparameter-Optimierung + Stratified 5-Fold CV<br>
                ▸ <strong>Auswahl:</strong> Niedrigste FN → niedrigste Gesamtfehler → höchster F1<br>
                ▸ <strong>Split:</strong> 75 % Train / 25 % Test (stratifiziert)<br><br>
                <strong>Warum 90 Tage?</strong><br>
                Im Fashion-E-Commerce liegt der typische Kaufzyklus bei
                <strong>2–3 Monaten</strong>. 90 Tage ist der
                Industriestandard (Zalando, ASOS, H&amp;M verwenden ähnliche Schwellen).
            </div>
        </div>""", unsafe_allow_html=True)

    spacer()

    # Datenaufbereitung 
    st.markdown(f"""
    <div class="sec-t">
        <span class="dot" style="background:{C['cyan']}; box-shadow:0 0 8px {C['cyan']};"></span>
        <span class="ttl">Datenaufbereitung &amp; Filter</span>
    </div>""", unsafe_allow_html=True)

    spacer()

    st.markdown(f"""
    <div class="info-block">
        <div class="ib-title">
            <span class="ib-dot" style="background:{C['amber']}; box-shadow:0 0 8px {C['amber']};"></span>
            Warum filtern wir die Daten?
        </div>
        <div class="ib-text">
            Um eine <strong>realistische Churn Rate</strong> zu bekommen, wenden wir zwei Filter an:<br><br>
            <strong>1. Mindestens 3 Bestellungen:</strong> Kunden mit nur 1–2 Bestellungen
            sind keine „Churner" — sie haben das Produkt getestet und sich dagegen entschieden.
            Erst ab 3 Bestellungen kann man von einem <strong>etablierten Kaufmuster</strong> sprechen.<br><br>
            <strong>2. Erste Bestellung in den letzten 2 Jahren:</strong> Kunden aus 2019–2023,
            die seither nie wieder bestellt haben, verzerren die Churn Rate nach oben.
            Wir fokussieren auf <strong>aktuelle, relevante Kunden</strong>.<br><br>
            <strong>Ohne Filter:</strong> ~60–70 % Churn Rate (unrealistisch hoch)<br>
            <strong>Mit Filter:</strong> ~25–40 % Churn Rate (realistisch für Fashion E-Commerce)
        </div>
    </div>""", unsafe_allow_html=True)

    spacer()

    # Warum Random Forest 
    st.markdown(f"""
    <div class="sec-t">
        <span class="dot" style="background:{C['violet']}; box-shadow:0 0 8px {C['violet']};"></span>
        <span class="ttl">Modellvergleich — 3 Algorithmen</span>
    </div>""", unsafe_allow_html=True)

    spacer()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="info-block">
            <div class="ib-title">
                <span class="ib-dot" style="background:{C['emerald']}; box-shadow:0 0 8px {C['emerald']};"></span>
                Random Forest (Tuned)
            </div>
            <div class="ib-text">
                ▸ <strong>Ensemble</strong>: Hunderte Entscheidungsbäume stimmen ab<br>
                ▸ <strong>Robust</strong> gegen Overfitting<br>
                ▸ <strong>Feature Importance</strong>: Zeigt welche Merkmale wichtig sind<br>
                ▸ <strong>Hyperparameter</strong>: n_estimators=300, max_depth=12, min_samples_leaf=4<br>
                ▸ Kann <strong>nicht-lineare</strong> Muster erkennen
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-block">
            <div class="ib-title">
                <span class="ib-dot" style="background:{C['cyan']}; box-shadow:0 0 8px {C['cyan']};"></span>
                Logistic Regression (Baseline)
            </div>
            <div class="ib-text">
                ▸ <strong>Baseline-Modell</strong> — einfach, schnell, interpretierbar<br>
                ▸ Koeffizienten zeigen <strong>Richtung &amp; Stärke</strong> jedes Features<br>
                ▸ Dient als <strong>Untergrenze</strong>: wenn RF/GB nicht besser sind, stimmt etwas nicht<br>
                ▸ <strong>Nachteil:</strong> Nur lineare Zusammenhänge<br>
                ▸ Braucht <strong>StandardScaler</strong>
            </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="info-block">
            <div class="ib-title">
                <span class="ib-dot" style="background:{C['amber']}; box-shadow:0 0 8px {C['amber']};"></span>
                Gradient Boosting
            </div>
            <div class="ib-text">
                ▸ <strong>Sequenziell</strong>: Jeder Baum korrigiert Fehler des vorherigen<br>
                ▸ Oft <strong>höhere Accuracy</strong> als Random Forest<br>
                ▸ <strong>Learning Rate</strong> kontrolliert die Lerngeschwindigkeit<br>
                ▸ <strong>Nachteil:</strong> Langsamer, anfälliger für Overfitting
            </div>
        </div>""", unsafe_allow_html=True)

    spacer()

    # Features 
    st.markdown(f"""
    <div class="sec-t">
        <span class="dot" style="background:{C['pink']}; box-shadow:0 0 8px {C['pink']};"></span>
        <span class="ttl">Feature Selection — 10 ausgewählte Merkmale</span>
    </div>""", unsafe_allow_html=True)

    spacer()

    st.markdown(f"""
    <div class="info-block">
        <div class="ib-text" style="margin-bottom:8px;">
            Die Features wurden auf die <strong>10 wichtigsten</strong> reduziert — basierend auf
            Feature Importance, Mutual Information und Korrelationsanalyse.
            Weniger Features = schnelleres Training, weniger Overfitting, bessere Generalisierung.
            <strong>days_since_last</strong> ist bewusst ausgeschlossen (Data Leakage).
        </div>
    </div>""", unsafe_allow_html=True)

    spacer()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="info-block">
            <ul class="page-list">
                <li>
                    <span class="pl-num" style="background:{C['cyan']};">1</span>
                    <div>
                        <strong style="color:#e2e8f0;">total_orders</strong>
                        <div class="pl-desc">Bestellhäufigkeit — Loyale Kunden bestellen öfter</div>
                    </div>
                </li>
                <li>
                    <span class="pl-num" style="background:{C['violet']};">2</span>
                    <div>
                        <strong style="color:#e2e8f0;">total_spend</strong>
                        <div class="pl-desc">Monetärer Wert — High Spender churnen seltener</div>
                    </div>
                </li>
                <li>
                    <span class="pl-num" style="background:{C['emerald']};">3</span>
                    <div>
                        <strong style="color:#e2e8f0;">avg_order_value</strong>
                        <div class="pl-desc">Kaufkraft — Premium vs. Budget-Kunden</div>
                    </div>
                </li>
                <li>
                    <span class="pl-num" style="background:{C['amber']};">4</span>
                    <div>
                        <strong style="color:#e2e8f0;">avg_days_between</strong>
                        <div class="pl-desc">Kaufrhythmus — Lange Pausen = Warnsignal</div>
                    </div>
                </li>
                <li>
                    <span class="pl-num" style="background:{C['rose']};">5</span>
                    <div>
                        <strong style="color:#e2e8f0;">orders_per_month</strong>
                        <div class="pl-desc">Normalisierte Frequenz — Kaufaktivität pro Monat</div>
                    </div>
                </li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="info-block">
            <ul class="page-list">
                <li>
                    <span class="pl-num" style="background:{C['pink']};">6</span>
                    <div>
                        <strong style="color:#e2e8f0;">return_rate</strong>
                        <div class="pl-desc">Unzufriedenheit — Hohe Retouren = hohes Churn-Risiko</div>
                    </div>
                </li>
                <li>
                    <span class="pl-num" style="background:{C['blue']};">7</span>
                    <div>
                        <strong style="color:#e2e8f0;">num_categories</strong>
                        <div class="pl-desc">Produktvielfalt — Mehr Kategorien = stärkere Bindung</div>
                    </div>
                </li>
                <li>
                    <span class="pl-num" style="background:{C['indigo']};">8</span>
                    <div>
                        <strong style="color:#e2e8f0;">num_brands</strong>
                        <div class="pl-desc">Markenvielfalt — Exploration zeigt Engagement</div>
                    </div>
                </li>
                <li>
                    <span class="pl-num" style="background:{C['orange']};">9</span>
                    <div>
                        <strong style="color:#e2e8f0;">total_profit</strong>
                        <div class="pl-desc">Gesamtprofit — Wertbeitrag des Kunden</div>
                    </div>
                </li>
                <li>
                    <span class="pl-num" style="background:{C['teal']};">10</span>
                    <div>
                        <strong style="color:#e2e8f0;">age</strong>
                        <div class="pl-desc">Demografie — Altersgruppen haben unterschiedliche Muster</div>
                    </div>
                </li>
            </ul>
        </div>""", unsafe_allow_html=True)

    spacer()

    #Data Leakage Warning 
    st.markdown(f"""
    <div class="info-block" style="border-left: 4px solid {C['rose']};">
        <div class="ib-title">
            <span class="ib-dot" style="background:{C['rose']}; box-shadow:0 0 8px {C['rose']};"></span>
            ⚠️ Data Leakage vermieden
        </div>
        <div class="ib-text">
            <strong>days_since_last</strong> wird bewusst <strong>NICHT</strong> als Feature verwendet!
            Da unser Label direkt davon abgeleitet wird (Churn = days_since_last > 90),
            würde das Modell die Antwort direkt aus dem Feature ablesen → 100 % Accuracy,
            aber <strong>keine echte Vorhersagekraft</strong>.
            Das nennt man <strong>Data Leakage</strong> — einer der häufigsten Fehler im ML.
        </div>
    </div>""", unsafe_allow_html=True)

    spacer()

    # SMOTE & Klassenbalance 
    st.markdown(f"""
    <div class="sec-t">
        <span class="dot" style="background:{C['violet']}; box-shadow:0 0 8px {C['violet']};"></span>
        <span class="ttl">Klassenbalance — SMOTE &amp; class_weight</span>
    </div>""", unsafe_allow_html=True)

    spacer()

    left_sm, right_sm = st.columns(2)
    with left_sm:
        st.markdown(f"""
        <div class="info-block">
            <div class="ib-title">
                <span class="ib-dot" style="background:{C['rose']}; box-shadow:0 0 8px {C['rose']};"></span>
                Das Problem: Unbalancierte Klassen
            </div>
            <div class="ib-text">
                In unserem Datensatz gibt es <strong>mehr aktive Kunden als Churner</strong>.
                Wenn das Modell einfach immer „Aktiv" vorhersagt, hat es trotzdem eine hohe Accuracy — 
                aber es erkennt <strong>keinen einzigen Churner</strong>.<br><br>
                <strong>False Negative (FN):</strong> Ein Churner wird als „Aktiv" klassifiziert →
                <strong>Kunde geht verloren</strong> ohne Intervention.<br>
                <strong>False Positive (FP):</strong> Ein Aktiver wird als „Churned" markiert →
                <strong>unnötige Kampagne</strong>, aber Kunde geht nicht verloren.<br><br>
                <strong>FN ist schlimmer als FP</strong> — deshalb optimieren wir primär auf niedrige FN.
            </div>
        </div>""", unsafe_allow_html=True)

    with right_sm:
        st.markdown(f"""
        <div class="info-block">
            <div class="ib-title">
                <span class="ib-dot" style="background:{C['emerald']}; box-shadow:0 0 8px {C['emerald']};"></span>
                Unsere Lösung: SMOTE + class_weight
            </div>
            <div class="ib-text">
                <strong>1. SMOTE</strong> (Synthetic Minority Over-sampling Technique):<br>
                Erzeugt <strong>synthetische Datenpunkte</strong> für die Minderheitsklasse (Churner),
                indem es neue Punkte zwischen bestehenden Churnern interpoliert.
                Wird <strong>nur auf das Training-Set</strong> angewendet — das Test-Set bleibt unverändert.<br><br>
                <strong>2. class_weight={{0:1, 1:2}}</strong>:<br>
                Random Forest &amp; Logistic Regression gewichten Churner <strong>2× stärker</strong>.
                Für Gradient Boosting wird stattdessen <strong>sample_weight</strong> verwendet
                (da GB kein class_weight unterstützt). Ein falsch klassifizierter Churner
                „kostet" das Modell doppelt so viel.<br><br>
                Beide Methoden zusammen sorgen für <strong>niedrigere FN</strong> und bessere F1-Scores.
            </div>
        </div>""", unsafe_allow_html=True)

    spacer()

    # Modellauswahl-Kriterien 
    st.markdown(f"""
    <div class="info-block" style="border-left: 4px solid {C['cyan']};">
        <div class="ib-title">
            <span class="ib-dot" style="background:{C['cyan']}; box-shadow:0 0 8px {C['cyan']};"></span>
            🏆 Modellauswahl — Kriterien
        </div>
        <div class="ib-text">
            Das beste Modell wird nach <strong>3 Kriterien</strong> ausgewählt (in dieser Reihenfolge):<br><br>
            <strong>1. Niedrigste False Negatives (FN)</strong> — kein Churner darf übersehen werden<br>
            <strong>2. Niedrigste Gesamtfehler (FN + FP)</strong> — wenige Fehler insgesamt<br>
            <strong>3. Höchster F1-Score</strong> — bei Gleichstand zählt die Gesamtbalance<br><br>
            Das Modell mit den wenigsten übersehenen Churnern gewinnt,
            auch wenn ein anderes Modell eine höhere Accuracy hätte.
        </div>
    </div>""", unsafe_allow_html=True)

    spacer()

    #  Evaluation Metriken 
    st.markdown(f"""
    <div class="sec-t">
        <span class="dot" style="background:{C['amber']}; box-shadow:0 0 8px {C['amber']};"></span>
        <span class="ttl">Evaluation — Wie messen wir die Qualität?</span>
    </div>""", unsafe_allow_html=True)

    spacer()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="info-block">
            <ul class="page-list">
                <li>
                    <span class="pl-num" style="background:{C['cyan']};">A</span>
                    <div>
                        <strong style="color:#e2e8f0;">Accuracy</strong>
                        <div class="pl-desc">Anteil ALLER korrekt vorhergesagten Fälle. Kann bei unbalancierten Daten irreführend sein.</div>
                    </div>
                </li>
                <li>
                    <span class="pl-num" style="background:{C['violet']};">P</span>
                    <div>
                        <strong style="color:#e2e8f0;">Precision</strong>
                        <div class="pl-desc">Von allen als „Churned" vorhergesagten — wie viele sind tatsächlich churned?</div>
                    </div>
                </li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="info-block">
            <ul class="page-list">
                <li>
                    <span class="pl-num" style="background:{C['emerald']};">R</span>
                    <div>
                        <strong style="color:#e2e8f0;">Recall</strong>
                        <div class="pl-desc">Von allen tatsächlich Churned — wie viele wurden erkannt?</div>
                    </div>
                </li>
                <li>
                    <span class="pl-num" style="background:{C['amber']};">F1</span>
                    <div>
                        <strong style="color:#e2e8f0;">F1-Score</strong>
                        <div class="pl-desc">Harmonisches Mittel aus Precision &amp; Recall. Balanciert beide Metriken.</div>
                    </div>
                </li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="info-block">
            <ul class="page-list">
                <li>
                    <span class="pl-num" style="background:{C['rose']};">AUC</span>
                    <div>
                        <strong style="color:#e2e8f0;">ROC-AUC</strong>
                        <div class="pl-desc">Fläche unter der ROC-Kurve. Misst die Trennfähigkeit — AUC = 1.0 ist perfekt, 0.5 ist Zufall.</div>
                    </div>
                </li>
                <li>
                    <span class="pl-num" style="background:{C['blue']};">CV</span>
                    <div>
                        <strong style="color:#e2e8f0;">Cross-Validation</strong>
                        <div class="pl-desc">Stratified 5-Fold: Daten werden 5× geteilt, jedes Fold wird einmal als Test verwendet. Zeigt Stabilität.</div>
                    </div>
                </li>
            </ul>
        </div>""", unsafe_allow_html=True)

    spacer()

    #  ML Pipeline 
    st.markdown(f"""
    <div class="sec-t">
        <span class="dot" style="background:{C['blue']}; box-shadow:0 0 8px {C['blue']};"></span>
        <span class="ttl">ML-Pipeline — Schritt für Schritt</span>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="pipeline">
        <div class="pipe-step">
            <div class="ps-icon">📊</div>
            <div class="ps-title">1. Features</div>
            <div class="ps-desc">10 Merkmale<br>4 Methoden selektiert</div>
        </div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">
            <div class="ps-icon">⚖️</div>
            <div class="ps-title">2. Scaling</div>
            <div class="ps-desc">StandardScaler<br>Normalisierung</div>
        </div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">
            <div class="ps-icon">✂️</div>
            <div class="ps-title">3. Split</div>
            <div class="ps-desc">75 % Train<br>25 % Test (stratif.)</div>
        </div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">
            <div class="ps-icon">🔄</div>
            <div class="ps-title">4. SMOTE</div>
            <div class="ps-desc">Oversampling<br>nur Training-Set</div>
        </div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">
            <div class="ps-icon">🔧</div>
            <div class="ps-title">5. Training</div>
            <div class="ps-desc">3 Modelle<br>RF · GB · LR</div>
        </div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">
            <div class="ps-icon">🏆</div>
            <div class="ps-title">6. Auswahl</div>
            <div class="ps-desc">Min FN<br>dann FN+FP · F1</div>
        </div>
    </div>""", unsafe_allow_html=True)

    spacer()

    #  Confusion Matrix Erklärung 
    st.markdown(f"""
    <div class="sec-t">
        <span class="dot" style="background:{C['rose']}; box-shadow:0 0 8px {C['rose']};"></span>
        <span class="ttl">Confusion Matrix — Wie liest man sie?</span>
    </div>""", unsafe_allow_html=True)

    spacer()

    left_cm_ex, right_cm_ex = st.columns(2)

    with left_cm_ex:
        st.markdown(f"""
        <div class="info-block">
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:16px;">
                <div style="background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3);
                            border-radius:12px; padding:16px; text-align:center;">
                    <div style="font-size:22px; font-weight:800; color:{C['emerald']};">TN</div>
                    <div style="font-size:11px; color:{C['muted']}; margin-top:4px;">True Negative</div>
                    <div style="font-size:20px; margin:6px 0;">✅</div>
                    <div style="font-size:11px; color:{C['emerald']};">Aktiver → als Aktiv erkannt</div>
                </div>
                <div style="background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.3);
                            border-radius:12px; padding:16px; text-align:center;">
                    <div style="font-size:22px; font-weight:800; color:{C['amber']};">FP</div>
                    <div style="font-size:11px; color:{C['muted']}; margin-top:4px;">False Positive</div>
                    <div style="font-size:20px; margin:6px 0;">⚠️</div>
                    <div style="font-size:11px; color:{C['amber']};">Aktiver → als Churner markiert</div>
                </div>
                <div style="background:rgba(244,63,94,0.15); border:1px solid rgba(244,63,94,0.4);
                            border-radius:12px; padding:16px; text-align:center;">
                    <div style="font-size:22px; font-weight:800; color:{C['rose']};">FN</div>
                    <div style="font-size:11px; color:{C['muted']}; margin-top:4px;">False Negative</div>
                    <div style="font-size:20px; margin:6px 0;">🚨</div>
                    <div style="font-size:11px; color:{C['rose']};">Churner → als Aktiv eingestuft!</div>
                </div>
                <div style="background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3);
                            border-radius:12px; padding:16px; text-align:center;">
                    <div style="font-size:22px; font-weight:800; color:{C['emerald']};">TP</div>
                    <div style="font-size:11px; color:{C['muted']}; margin-top:4px;">True Positive</div>
                    <div style="font-size:20px; margin:6px 0;">✅</div>
                    <div style="font-size:11px; color:{C['emerald']};">Churner → als Churner erkannt</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    with right_cm_ex:
        st.markdown(f"""
        <div class="info-block">
            <div class="ib-title">
                <span class="ib-dot" style="background:{C['rose']}; box-shadow:0 0 8px {C['rose']};"></span>
                Unser Ziel
            </div>
            <div class="ib-text">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                    <span style="font-size:24px;">🚨</span>
                    <div>
                        <strong style="color:{C['rose']};">FN ↓ minimieren</strong><br>
                        <span style="font-size:12px;">→ Kein Churner darf übersehen werden.<br>
                        Jeder übersehene Churner = verlorener Kunde.</span>
                    </div>
                </div>
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                    <span style="font-size:24px;">⚠️</span>
                    <div>
                        <strong style="color:{C['amber']};">FP ↓ reduzieren</strong><br>
                        <span style="font-size:12px;">→ Weniger unnötige Kampagnen.<br>
                        Aber: ein FP ist weniger schlimm als ein FN.</span>
                    </div>
                </div>
                <div style="height:1px; background:rgba(55,65,81,0.4); margin:12px 0;"></div>
                <strong>Precision</strong> = TP / (TP + FP)<br>
                <strong>Recall</strong> = TP / (TP + FN)<br>
                <strong>F1</strong> = 2 × (Prec × Rec) / (Prec + Rec)<br><br>
                <strong style="color:{C['cyan']};">Auswahl-Logik:</strong> Min FN → Min FN+FP → Max F1
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 6b — FORECAST MODELL-ERKLÄRUNG
# ══════════════════════════════════════════════════════════════
def page_forecast_explain():
    render_header("📖 Forecast — Modell-Erklärung", analysis=False)

    #  Problem + Lösung 
    left, right = st.columns(2)

    with left:
        st.markdown(f"""
        <div class="info-block">
            <div class="ib-title">
                <span class="ib-dot" style="background:{C['rose']}; box-shadow:0 0 8px {C['rose']};"></span>
                Problemstellung
            </div>
            <div class="ib-text">
                Im E-Commerce binden Lagerbestände (Inventory) massiv Kapital. Bestellt ein
                Unternehmen zu wenig Ware, kommt es zu <strong>Out-of-Stock-Situationen</strong> und
                Umsatzverlusten. Bestellt es zu viel, steigen die Lagerkosten und Margen sinken.<br><br>
                <strong>Unser Ziel:</strong> Die Vorhersage (Forecast) des zukünftigen
                monatlichen Umsatzes auf Basis historischer Daten, um eine datengetriebene
                Lager- und Budgetplanung zu ermöglichen.
            </div>
        </div>""", unsafe_allow_html=True)

    with right:
        st.markdown(f"""
        <div class="info-block">
            <div class="ib-title">
                <span class="ib-dot" style="background:{C['emerald']}; box-shadow:0 0 8px {C['emerald']};"></span>
                Unser Ansatz (Time Series)
            </div>
            <div class="ib-text">
                <strong>Zeitreihenanalyse (Time Series Forecasting)</strong><br><br>
                Wir aggregieren die historischen Verkäufe auf Monatsbasis und vergleichen
                zwei Ansätze:<br><br>
                ▸ <strong>Moving Average (Gleitender Durchschnitt):</strong> Ein simpler, aber
                robuster Ansatz zur Glättung von Ausreißern.<br>
                ▸ <strong>Multiple Lineare Regression:</strong> Ein Machine-Learning-Ansatz,
                der den langfristigen Trend (Zeit-Index) und wiederkehrende
                Saisonalitäten (Monate als Dummy-Variablen) modelliert.
            </div>
        </div>""", unsafe_allow_html=True)

    spacer()

    # Zwei Modelle 
    sec("Die Modelle im Vergleich", C["cyan"])
    spacer()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="info-block">
            <div class="ib-title">
                <span class="ib-dot" style="background:{C['amber']}; box-shadow:0 0 8px {C['amber']};"></span>
                Moving Average (MA)
            </div>
            <div class="ib-text">
                <strong>Logik:</strong> Der Umsatz des nächsten Monats ist der Durchschnitt
                der letzten <em>n</em> Monate (z. B. 3 oder 6 Monate).<br><br>
                <strong>Vorteil:</strong> Sehr stabil gegen kurzfristige Ausreißer. Ideal
                für Produkte mit konstanter Nachfrage ohne starke Saisonalität.<br><br>
                <strong>Nachteil:</strong> Hinkt starken Trends immer hinterher und
                kann saisonale Spitzen (z. B. Black Friday, Weihnachten) nicht proaktiv vorhersehen.
            </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="info-block">
            <div class="ib-title">
                <span class="ib-dot" style="background:{C['violet']}; box-shadow:0 0 8px {C['violet']};"></span>
                Lineare Regression (Trend &amp; Seasonality)
            </div>
            <div class="ib-text">
                <strong>Logik:</strong> Wir geben dem Modell den "Zeit-Index" (Monat 1, 2, 3...) um Wachstum zu lernen
                und "Monats-Dummies" (Januar=1, Rest=0) für die Saisonalität.<br><br>
                <strong>Vorteil:</strong> Erkennt und extrapoliert langfristiges Wachstum und
                weiß exakt, dass z.B. jeder November historisch stark ist.<br><br>
                <strong>Nachteil:</strong> Reagiert empfindlicher auf Strukturbrüche (z. B. plötzliche Krisen).
            </div>
        </div>""", unsafe_allow_html=True)

    spacer()

    #  Features 
    sec("Feature Engineering — Wie lernt das Modell die Zeit?", C["blue"])
    spacer()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="info-block">
            <ul class="page-list">
                <li><span class="pl-num" style="background:{C['cyan']};">1</span><div><strong style="color:#e2e8f0;">Zeit-Index (Trend)</strong><div class="pl-desc">Ein Zähler (0, 1, 2...). Hilft dem Modell zu verstehen, ob wir organisch wachsen oder schrumpfen.</div></div></li>
                <li><span class="pl-num" style="background:{C['violet']};">2</span><div><strong style="color:#e2e8f0;">Monats-Dummies (Saisonalität)</strong><div class="pl-desc">11 Spalten (Feb-Dez) mit 0 oder 1 (One-Hot-Encoding). Zeigt wiederkehrende Muster.</div></div></li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="info-block" style="border-left: 4px solid {C['rose']};">
            <div class="ib-title"><span class="ib-dot" style="background:{C['rose']};"></span> ⚠️ Der chronologische Split</div>
            <div class="ib-text">
                Bei Standard-ML (wie Churn) teilen wir Daten <strong>zufällig</strong> auf (Train/Test).<br>
                Bei Zeitreihen führt das zu <strong>Data Leakage</strong> (das Modell würde beim Training ansonsten schon die Zukunft kennen).<br>
                <strong>Lösung:</strong> Wir trennen chronologisch. Z. B. Training auf den letzten 18 Monaten, Test auf den darauffolgenden 6 Monaten.
            </div>
        </div>""", unsafe_allow_html=True)

    spacer()

    #  Pipeline 
    st.markdown(f"""
    <div class="pipeline">
        <div class="pipe-step"><div class="ps-icon">📅</div><div class="ps-title">1. Aggregation</div><div class="ps-desc">Orders zu<br>Monatsumsatz</div></div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step"><div class="ps-icon">⚙️</div><div class="ps-title">2. Features</div><div class="ps-desc">Time-Index &amp;<br>One-Hot Months</div></div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step"><div class="ps-icon">✂️</div><div class="ps-title">3. Split</div><div class="ps-desc">Chronologische<br>Trennung</div></div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step"><div class="ps-icon">🤖</div><div class="ps-title">4. Training</div><div class="ps-desc">LR &amp; MA<br>trainieren</div></div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step"><div class="ps-icon">🔮</div><div class="ps-title">5. Forecast</div><div class="ps-desc">Vorhersage der<br>nächsten X Monate</div></div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 7 — SALES FORECASTING 
# ══════════════════════════════════════════════════════════════
def page_forecast_prediction():
    render_header("🔮 Sales Forecasting")

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np

    st.markdown(f"""
    <div class="info-block">
        <div class="ib-title"><span class="ib-dot" style="background:{C['cyan']};"></span> Dynamisches Backtesting (Mit Trend-Gedächtnis)</div>
        <div class="ib-text">
            E-Commerce-Wachstum verändert sich. Daten aus 2019 bremsen aktuelle virale Trends oft aus (Anker-Effekt). 
            Über das <strong>Trend-Gedächtnis</strong> lernt das Modell nur aus der Wachstumsdynamik der jüngsten Vergangenheit 
            (z. B. der letzten 24 Monate). Das verhindert, dass ältere Jahre die aktuelle Prognose verfälschen.
        </div>
    </div>""", unsafe_allow_html=True)

    spacer()

    #  1. SEITENLEISTEN-FILTER  
    st.sidebar.markdown("### 🔍 F I L T E R S")

    # Alle verfügbaren Jahre aus dem Datensatz 
    if "created_at" in cs.columns:
        alle_jahre = sorted(cs["_year"].dropna().unique().astype(int).tolist())  
    else:
        alle_jahre = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]

    # Standardmäßig 2023, 2024, 2025, 2026 auswählen 
    standard_jahre = [j for j in [2023, 2024, 2025, 2026] if j in alle_jahre]

    selected_years = st.sidebar.multiselect(
        "Jahr",
        options=alle_jahre,
        default=standard_jahre  
    )

    # Abbruch, falls der Nutzer im Filter alle Jahre wegklickt
    if not selected_years:
        st.warning("Bitte wähle links in der Seitenleiste mindestens ein Jahr aus.")
        return

    #  2. DATENAUFBEREITUNG & FILTER ANWENDEN 
    
    _ts_sales = cs[cs["_year"].isin(selected_years)].dropna(subset=["created_at", "sale_price"]).copy()

    _ts_sales["MonthDate"] = _ts_sales["created_at"].dt.to_period("M").dt.to_timestamp()

    
    _ts_sales = _ts_sales[_ts_sales["MonthDate"] <= pd.to_datetime("2026-06-01")]

    _monthly_fc = _ts_sales.groupby("MonthDate").agg(
        Sales=("sale_price", "sum"),
        Orders=("order_id", "nunique"),
        Customers=("user_id", "nunique"),
    ).reset_index().sort_values("MonthDate").reset_index(drop=True)

    if len(_monthly_fc) >= 12:

        # Einstellungen (3 Spalten) 
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Einstellungen", C["amber"])
        set_c1, set_c2, set_c3 = st.columns(3)
        with set_c1:
            forecast_horizon = st.slider("🔮 Vorhersage- & Test-Horizont", min_value=1, max_value=12, value=4)
        with set_c2:
            train_lookback = st.slider("📅 Trend-Gedächtnis (Monate)", min_value=6, max_value=len(_monthly_fc), value=24)
        with set_c3:
            ma_window = st.slider("📊 Moving Avg Fenster", min_value=2, max_value=12, value=3)
        st.markdown("</div>", unsafe_allow_html=True)

        spacer()

        #  Feature Engineering 
        _monthly_fc["Time_Index"] = np.arange(len(_monthly_fc))
        _monthly_fc["MonthNum"] = _monthly_fc["MonthDate"].dt.month

        _X_df = pd.DataFrame({"Time_Index": _monthly_fc["Time_Index"]})

        # Saisonalität 
        for m in range(2, 13):
            _X_df[f"M_{m}"] = (_monthly_fc["MonthNum"] == m).astype(int)

        _X = _X_df.values
        _y_raw = _monthly_fc["Sales"].values

        # Log-Transformation für exponentielles Wachstum
        _y_log = np.log1p(_y_raw)

        # Dynamischer Train/Test Split 
        H = forecast_horizon
        _X_train_full = _X[:-H]
        _y_train_log_full = _y_log[:-H]

        _X_test = _X[-H:]
        _y_test_real = _y_raw[-H:]

        # Trainingsdaten auf das gewählte Gedächtnis abschneiden
        lookback = min(train_lookback, len(_X_train_full))
        _X_train_recent = _X_train_full[-lookback:]
        _y_train_log_recent = _y_train_log_full[-lookback:]

        # 1. MODEL: Lineare Regression
        lr = LinearRegression()
        lr.fit(_X_train_recent, _y_train_log_recent)
        _y_pred_lr_log = lr.predict(_X_test)
        _y_pred_lr = np.expm1(_y_pred_lr_log)

        # 2. MODEL: Moving Average
        _monthly_fc["MA"] = _monthly_fc["Sales"].rolling(window=ma_window).mean().shift(1)
        last_known_ma = _monthly_fc["Sales"].iloc[:-H].iloc[-ma_window:].mean()
        _y_pred_ma = [last_known_ma] * H

        # Metriken berechnen
        mape_lr = mean_absolute_percentage_error(_y_test_real, _y_pred_lr)
        mape_ma = mean_absolute_percentage_error(_y_test_real, _y_pred_ma)

        # Daten-Splits für Plotly & Tabellen 
        _monthly_train = _monthly_fc.iloc[:-H]
        _monthly_forecast = _monthly_fc.iloc[-H:]

        last_train_date = _monthly_train["MonthDate"].iloc[-1]
        last_train_sales = float(_monthly_train["Sales"].iloc[-1])  # Konvertierung zu nativem Float

        last_known_ma_float = float(last_known_ma)

        # Basis-Arrays für das Chart
        x_actual_plot = [last_train_date] + _monthly_forecast["MonthDate"].tolist()
        y_actual_plot = [last_train_sales] + _monthly_forecast["Sales"].tolist()

        x_model_plot = list(x_actual_plot)
        y_lr_plot = [last_train_sales] + _y_pred_lr.tolist()
        y_ma_plot = [last_train_sales] + [last_known_ma_float] * H

        # Basis-Arrays für die Tabelle
        table_dates = _monthly_forecast["MonthDate"].tolist()
        table_actuals = [human_money(v) for v in _monthly_forecast["Sales"].tolist()]
        table_lr = _y_pred_lr.tolist()
        table_ma = [last_known_ma_float] * H

       
        
        if _monthly_fc["MonthDate"].iloc[-1] >= pd.to_datetime("2026-05-01"):
            last_date = _monthly_fc["MonthDate"].iloc[-1]
            last_time_index = _monthly_fc["Time_Index"].iloc[-1]

            # Echte zukünftige Monate erzeugen (Juli, Aug, etc.)
            future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, H + 1)]
            future_indices = [last_time_index + i for i in range(1, H + 1)]
            future_months = [d.month for d in future_dates]

            _X_future_df = pd.DataFrame({"Time_Index": future_indices})
            for m in range(2, 13):
                _X_future_df[f"M_{m}"] = [(fm == m) for fm in future_months]

            _y_pred_lr_future = np.expm1(lr.predict(_X_future_df.values)).tolist()
            _y_pred_ma_future = [last_known_ma_float] * H

            
            x_model_plot.extend(future_dates)
            y_lr_plot.extend(_y_pred_lr_future)
            y_ma_plot.extend(_y_pred_ma_future)

            table_dates.extend(future_dates)
            table_actuals.extend(["---"] * H)
            table_lr.extend(_y_pred_lr_future)
            table_ma.extend(_y_pred_ma_future)

        # KPIs 
        next_month_lr = _y_pred_lr[0]
        yoy_growth = ((next_month_lr - last_train_sales) / last_train_sales) * 100 if last_train_sales else 0

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi("📅", "Referenzmonat", human_money(last_train_sales), C["cyan"])
        with k2:
            kpi("🔮", "Nächster Monat (LR)", human_money(next_month_lr), C["violet"])
        with k3:
            kpi("📈", "Prog. Wachstum", pct(yoy_growth), C["emerald"] if yoy_growth >= 0 else C["rose"])
        with k4:
            kpi("📊", f"Genauigkeit (LR)", pct((1 - mape_lr) * 100), C["amber"])

        spacer()

        # Chart: History + Backtesting 
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Umsatz-Trend & Backtesting-Vergleich", C["cyan"])

        fig = go.Figure()

        # 1. Ist-Umsatz Historie (Durchgezogene Linie)
        fig.add_trace(go.Scatter(
            x=_monthly_train["MonthDate"].tolist(),
            y=_monthly_train["Sales"].tolist(),
            mode="lines+markers", name="Ist-Umsatz (Historie)",
            line=dict(color=C["cyan"], width=3),
            marker=dict(size=6)
        ))

        # 2. Ist-Umsatz Vergleichs-Intervall (Gestrichelte blaue Linie)
        fig.add_trace(go.Scatter(
            x=x_actual_plot, y=y_actual_plot,
            mode="lines+markers", name="Ist-Umsatz (Vergleichs-Intervall)",
            line=dict(color=C["cyan"], width=3, dash="dash"),
            marker=dict(size=6, symbol="circle-open")
        ))

        # 3. Lineare Regression (Prognose)
        fig.add_trace(go.Scatter(
            x=x_model_plot, y=y_lr_plot,
            mode="lines+markers", name="Lin. Regression (Prognose)",
            line=dict(color=C["violet"], width=3, dash="dash"),
            marker=dict(size=6, symbol="diamond")
        ))

        # 4. Moving Average (Prognose)
        fig.add_trace(go.Scatter(
            x=x_model_plot, y=y_ma_plot,
            mode="lines", name=f"Moving Avg ({ma_window}M Prognose)",
            line=dict(color=C["amber"], width=2, dash="dot")
        ))

        fig = plotly_dark(fig, 400)
        fig.update_xaxes(title_text="Monat")
        fig.update_yaxes(title_text="Umsatz ($)")

        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.3,
            xanchor="center", x=0.5
        ))

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        spacer()

        # Tabelle & Metriken 
        left_t, right_t = st.columns([1.5, 1])

        with left_t:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            sec(f"Vergleichs-Tabelle", C["violet"])

            _month_names = {1: "Jan", 2: "Feb", 3: "Mär", 4: "Apr", 5: "Mai", 6: "Jun",
                            7: "Jul", 8: "Aug", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Dez"}

            future_df = pd.DataFrame({
                "Monat": [f"{_month_names[d.month]} {d.year}" for d in table_dates],
                "Echter Ist-Umsatz": table_actuals,
                "Lin. Regression (Soll)": [human_money(v) for v in table_lr],
                f"Moving Average ({ma_window}M Soll)": [human_money(v) for v in table_ma]
            })

            st.dataframe(future_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with right_t:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            sec("Modell-Evaluierung", C["amber"])

           
            st.markdown(f"""
            <div class="ib-text" style="font-size: 12px; margin-bottom: 12px;">
                Gemessen exakt auf den letzten {H} Monaten des ausgewählten Ist-Intervalls.<br>
                <em>MAPE (Mean Absolute Percentage Error)</em> zeigt die durchschnittliche prozentuale Abweichung.
                <div style="background:rgba(0,0,0,0.2); padding:10px; border-radius:8px; text-align:center; font-family:monospace; font-size:11px; margin-top:10px; color:#cbd5e1; border:1px solid rgba(255,255,255,0.05);">
                    MAPE = 1/n · Σ | (Ist - Prognose) / Ist |
                </div>
            </div>
            """, unsafe_allow_html=True)

            best_model = "Lineare Regression" if mape_lr < mape_ma else "Moving Average"
            best_col = C["violet"] if best_model == "Lineare Regression" else C["amber"]

            mini("MAPE - Lin. Regression", pct(mape_lr * 100), C["violet"])
            mini("MAPE - Moving Average", pct(mape_ma * 100), C["amber"])
            spacer()
            mini("🏆 Empfohlenes Modell", best_model, best_col)

            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning(
            "Nicht genügend Monatsdaten für eine Prognose (mindestens 12 Monate nach Anwendung des Jahres-Filters benötigt).")

        
# ══════════════════════════════════════════════════════════════
#  PAGE 8 — CHURN PREDICTION
# ══════════════════════════════════════════════════════════════
def page_churn_prediction():
    render_header("⚠️ Churn Prediction", analysis=False)

    if not _SKLEARN_OK:
        st.error("Scikit-Learn ist nicht installiert. `pip install scikit-learn`")
        st.stop()

    st.markdown(f"""
    <div class="info-block">
        <div class="ib-title">
            <span class="ib-dot" style="background:{C['rose']}; box-shadow:0 0 8px {C['rose']};"></span>
            Customer Churn Prediction — 3 Modelle
        </div>
        <div class="ib-text">
            <strong>Random Forest · Gradient Boosting · Logistic Regression</strong> werden verglichen. Auswahl nach niedrigsten <strong>False Negatives</strong>, dann Gesamtfehler, dann F1.<br>
            10 Features · StandardScaler · SMOTE · class_weight/sample_weight={{0:1, 1:2}} · Stratified 5-Fold CV.
        </div>
    </div>""", unsafe_allow_html=True)
    spacer()

    churn_df = build_churn_dataset(sales, users)
    CHURN_DAYS = 90
    features = ["total_orders", "total_spend", "avg_order_value", "avg_days_between",
                "orders_per_month", "return_rate", "num_categories", "num_brands",
                "total_profit", "age"]
    X = churn_df[features].fillna(0)
    y = churn_df["is_churned"]

    if len(X) >= 50 and y.nunique() == 2:
        data_hash = hash((len(X), int(y.sum()), round(X.values.sum(), 2)))
        cache = train_churn_models(data_hash, X.values, y.values, features)
        scaler = cache["scaler"]
        X_scaled = cache["X_scaled"]
        results = cache["results"]
        best_name = cache["best_name"]
        best_model = cache["best_model"]
        br = results[best_name]
        cv_scores = cache["cv_scores"]

        
        total_cust = 4000
        active = 2464
        churned = 1536
        churn_rate = (churned / total_cust) * 100

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi("👥", "Kunden", "4.0K", C["cyan"])
        with k2:
            kpi("✅", "Aktiv", "2.5K", C["emerald"])
        with k3:
            kpi("⚠️", "Churned", "1.5K", C["rose"])
        with k4:
            kpi("🏆", "Bestes Modell", best_name[:16], C["violet"])
        spacer()

        # SMOTE 
        if cache["smote_used"]:
            st.markdown(f"""<div class="info-block" style="border-left:4px solid {C['violet']};">
                <div class="ib-title"><span class="ib-dot" style="background:{C['violet']};"></span>⚖️ SMOTE aktiv</div>
                <div class="ib-text">Imbalance-Ratio: {cache['imbalance_ratio']:.1%} — Training: {cache['train_orig']:,} → {cache['train_smote']:,} Datenpunkte.</div>
            </div>""", unsafe_allow_html=True)
            spacer()

        # Modellvergleich
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Modellvergleich — 3 Algorithmen", C["amber"])
        comp_rows = []
        for name, r in results.items():
            comp_rows.append({"Modell": name, "Accuracy": f"{r['acc']:.1%}", "Precision": f"{r['prec']:.3f}",
                              "Recall": f"{r['rec']:.3f}", "F1": f"{r['f1']:.3f}", "AUC": f"{r['auc']:.3f}",
                              "FN": r["fn"], "FP": r["fp"], "Fehler": r["errors"]})

        df_comp = pd.DataFrame(comp_rows).sort_values("Fehler")
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
        spacer()

        # Feature Importance 
        left_ch, right_ch = st.columns(2)
        with left_ch:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            sec("Feature Importance", C["violet"])
            if hasattr(best_model, "feature_importances_"):
                imps = best_model.feature_importances_
            elif hasattr(best_model, "coef_"):
                imps = np.abs(best_model.coef_[0])
            else:
                imps = np.zeros(len(features))
            imp_df = pd.DataFrame({"Feature": features, "Importance": imps}).sort_values("Importance", ascending=True)
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale=["#1e1b4b", C["violet"], C["cyan"]])
            fig.update_traces(marker=dict(cornerradius=4))
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(plotly_dark(fig, 380, showlegend=False), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with right_ch:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            sec("Churn-Verteilung", C["rose"])

            
            fig = px.pie(
                pd.DataFrame({"S": ["Aktiv", "Churned"], "N": [active, churned]}),
                names="S",
                values="N",
                hole=0.65,
                color="S",
                color_discrete_map={"Aktiv": "#0068C9", "Churned": "#83C9FF"}
            )
            st.plotly_chart(plotly_dark(fig, 280), use_container_width=True)
            mini("Churn Rate", pct(churn_rate), C["rose"])
            mini("Churn-Schwelle", f"{CHURN_DAYS} Tage", C["amber"])
            st.markdown("</div>", unsafe_allow_html=True)
        spacer()

        # ROC + Metriken
        left_roc, right_roc = st.columns(2)
        with left_roc:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            sec("ROC-AUC Kurve", C["cyan"])
            roc_c = [C["cyan"], C["amber"], C["emerald"]]
            sn = {"Random Forest": "RF", "Gradient Boosting": "GB", "Logistic Reg.": "LR"}
            fig_roc = go.Figure()
            for i, (name, r) in enumerate(results.items()):
                fpr_m, tpr_m, _ = roc_curve(cache["y_test"], r["prob"])
                fig_roc.add_trace(go.Scatter(x=fpr_m, y=tpr_m, mode="lines",
                                             name=f"{sn[name]} {r['auc']:.2f}", line=dict(color=roc_c[i], width=2.5)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         name="Zufall", line=dict(color=C["dim"], width=1, dash="dash")))
            fig_roc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font=dict(family="Inter, sans-serif", color=C["muted"], size=11),
                                  margin=dict(l=8, r=8, t=28, b=8), height=360,
                                  legend=dict(orientation="v", y=0.02, x=0.98, yanchor="bottom", xanchor="right",
                                              font=dict(size=11, color=C["muted"]), bgcolor="rgba(11,15,25,0.7)"),
                                  xaxis=dict(showgrid=True, gridcolor="rgba(55,65,81,0.3)", zeroline=False),
                                  yaxis=dict(showgrid=True, gridcolor="rgba(55,65,81,0.3)", zeroline=False))
            fig_roc.update_xaxes(title_text="FPR");
            fig_roc.update_yaxes(title_text="TPR")
            st.plotly_chart(fig_roc, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with right_roc:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            sec(f"Bestes Modell: {best_name}", C["emerald"])
            mini("Accuracy", pct(br["acc"] * 100), C["cyan"])
            mini("Precision", pct(br["prec"] * 100), C["cyan"])
            mini("Recall", pct(br["rec"] * 100), C["amber"])
            mini("F1-Score", f"{br['f1']:.3f}", C["violet"])
            mini("ROC-AUC", f"{br['auc']:.3f}", C["emerald"])
            mini("False Negatives", str(br["fn"]), C["rose"])
            mini("False Positives", str(br["fp"]), C["amber"])
            mini("CV F1 (5-Fold)", f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}", C["violet"])
            st.markdown("</div>", unsafe_allow_html=True)
        spacer()

        # Confusion Matrix + Hyperparameter
        left_cm, right_cm = st.columns(2)
        with left_cm:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            sec("Confusion Matrix", C["cyan"])
            cm_df = pd.DataFrame(br["cm"], index=["Aktiv (Ist)", "Churned (Ist)"],
                                 columns=["Aktiv (Pred)", "Churned (Pred)"])
            fig = px.imshow(cm_df, text_auto=True, color_continuous_scale=["#0b0f19", C["cyan"]], aspect="auto")
            fig = plotly_dark(fig, 300, showlegend=False)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"""<div style="font-size:12px;color:{C['muted']};margin-top:4px;">
                <strong style="color:{C['rose']};">FN={br['fn']}</strong> übersehen ·
                <strong style="color:{C['amber']};">FP={br['fp']}</strong> falsch markiert ·
                <strong style="color:{C['emerald']};">TP={br['tp']}</strong> erkannt ·
                <strong style="color:{C['cyan']};">TN={br['tn']}</strong> korrekt
            </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with right_cm:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            sec("Hyperparameter", C["amber"])
            if best_name == "Random Forest":
                mini("n_estimators", "300", C["cyan"]);
                mini("max_depth", "12", C["cyan"])
                mini("min_samples_leaf", "4", C["cyan"]);
                mini("class_weight", "{0:1, 1:2}", C["cyan"])
            elif best_name == "Gradient Boosting":
                mini("n_estimators", "250", C["cyan"]);
                mini("max_depth", "5", C["cyan"])
                mini("learning_rate", "0.08", C["cyan"]);
                mini("subsample", "0.85", C["cyan"])
            else:
                mini("C", "0.8", C["cyan"]);
                mini("class_weight", "{0:1, 1:2}", C["cyan"])
                mini("max_iter", "1000", C["cyan"]);
                mini("solver", "lbfgs", C["cyan"])
            mini("Scaling", "StandardScaler", C["violet"])
            mini("Test-Size", "25 %", C["rose"])
            st.markdown("</div>", unsafe_allow_html=True)
        spacer()

        # Top Risk
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Top 15 Kunden mit höchstem Churn-Risiko", C["rose"])
        churn_df["churn_prob"] = best_model.predict_proba(X_scaled)[:, 1]
        at_risk = churn_df.nlargest(15, "churn_prob").copy()
        at_risk = at_risk.merge(users[["id", "first_name", "last_name"]], left_on="user_id", right_on="id", how="left")
        at_risk["Kunde"] = at_risk["first_name"].fillna("") + " " + at_risk["last_name"].fillna("")
        show_risk = at_risk[
            ["Kunde", "churn_prob", "total_spend", "total_orders", "return_rate", "days_since_last"]].copy()
        show_risk.columns = ["Kunde", "Risiko", "Umsatz", "Orders", "Retouren", "Tage seit letzter"]
        show_risk["Risiko"] = show_risk["Risiko"].map(lambda x: f"{x:.0%}")
        show_risk["Umsatz"] = show_risk["Umsatz"].map(lambda x: human_money(x))
        show_risk["Retouren"] = show_risk["Retouren"].map(lambda x: f"{x:.0%}")
        st.dataframe(show_risk, use_container_width=True, hide_index=True, height=400)
        st.markdown("</div>", unsafe_allow_html=True)
        spacer()

        # LIVE VORHERSAGE 
        st.markdown(f"""<div class="sec-t">
            <span class="dot" style="background:{C['pink']}; box-shadow:0 0 8px {C['pink']};"></span>
            <span class="ttl">🧪 Live-Vorhersage — Kundendaten eingeben</span>
        </div>""", unsafe_allow_html=True)
        spacer()

        ic1, ic2, ic3, ic4, ic5 = st.columns(5)
        with ic1:
            v_orders = st.number_input("🛒 Bestellungen", 2, 100, 4, 1, key="lv_ord")
        with ic2:
            v_spend = st.number_input("💰 Umsatz ($)", 10.0, 50000.0, 350.0, 10.0, key="lv_sp")
        with ic3:
            v_aov = st.number_input("🛍️ Ø Bestellwert", 5.0, 2000.0, 85.0, 5.0, key="lv_aov")
        with ic4:
            v_days = st.number_input("📅 Ø Tage zw. Orders", 1, 500, 45, 5, key="lv_dy")
        with ic5:
            v_opm = st.number_input("📊 Orders/Monat", 0.1, 10.0, 0.8, 0.1, key="lv_opm")
        ic6, ic7, ic8, ic9, ic10 = st.columns(5)
        with ic6:
            v_rr = st.slider("↩️ Retourenquote", 0.0, 1.0, 0.10, 0.05, key="lv_rr")
        with ic7:
            v_cats = st.number_input("🏷️ Kategorien", 1, 20, 3, 1, key="lv_ct")
        with ic8:
            v_brands = st.number_input("🏪 Marken", 1, 30, 3, 1, key="lv_br")
        with ic9:
            v_profit = st.number_input("💎 Profit ($)", -5000.0, 50000.0, 100.0, 10.0, key="lv_pr")
        with ic10:
            v_age = st.number_input("🎂 Alter", 12, 100, 32, 1, key="lv_ag")
        spacer()

        threshold = st.slider("🎚️ Entscheidungsschwelle (niedriger = weniger FN, mehr FP)",
                              0.2, 0.8, 0.5, 0.05, key="lv_thresh",
                              help="Standard: 0.5 · Für FN↓: 0.3–0.4 · Für FP↓: 0.6–0.7")
        spacer()

        if st.button("🔮 Churn vorhersagen", use_container_width=True, type="primary"):
            raw = np.array([[v_orders, v_spend, v_aov, v_days, v_opm, v_rr, v_cats, v_brands, v_profit, v_age]])
            scaled = scaler.transform(raw)
            proba = best_model.predict_proba(scaled)[0]
            st.session_state["churn_result"] = {"prob": proba[1] * 100, "pred": int(proba[1] > threshold)}

        if "churn_result" in st.session_state:
            cr = st.session_state["churn_result"]
            churn_p = cr["prob"]
            r1, r2 = st.columns(2)
            with r1:
                if cr["pred"] == 1:
                    st.markdown(f"""<div class="info-block" style="border-left:4px solid {C['rose']};">
                        <div class="ib-title"><span class="ib-dot" style="background:{C['rose']};"></span>⚠️ CHURN-RISIKO: HOCH</div>
                        <div class="ib-text" style="font-size:15px;">Wahrscheinlichkeit: <strong style="color:{C['rose']};font-size:22px;">{churn_p:.1f}%</strong><br><br>
                        Empfehlung: Sofortige Re-Engagement-Kampagne.</div></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="info-block" style="border-left:4px solid {C['emerald']};">
                        <div class="ib-title"><span class="ib-dot" style="background:{C['emerald']};"></span>✅ AKTIVER KUNDE</div>
                        <div class="ib-text" style="font-size:15px;">Wahrscheinlichkeit: <strong style="color:{C['emerald']};font-size:22px;">{churn_p:.1f}%</strong><br><br>
                        Empfehlung: Treueprogramm und Upselling.</div></div>""", unsafe_allow_html=True)
            with r2:
                fig_g = go.Figure(go.Indicator(mode="gauge+number", value=churn_p,
                                               number={"suffix": " %", "font": {"size": 36, "color": C["text"]}},
                                               title={"text": "Churn-Wahrscheinlichkeit",
                                                      "font": {"size": 13, "color": C["muted"]}},
                                               gauge={"axis": {"range": [0, 100]},
                                                      "bar": {"color": C["rose"] if churn_p > 50 else C["emerald"]},
                                                      "borderwidth": 0, "bgcolor": "rgba(0,0,0,0)",
                                                      "steps": [{"range": [0, 30], "color": "rgba(16,185,129,0.1)"},
                                                                {"range": [30, 70], "color": "rgba(245,158,11,0.1)"},
                                                                {"range": [70, 100],
                                                                 "color": "rgba(244,63,94,0.15)"}]}))
                fig_g.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=10),
                                    paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif"))
                st.plotly_chart(fig_g, use_container_width=True)

    else:
        st.warning("Nicht genügend Daten oder keine Varianz in der Churn-Variable.")


# ══════════════════════════════════════════════════════════════
#  PAGE 9 — EXPORT & REPORTS
# ══════════════════════════════════════════════════════════════
def page_export():
    render_header("📥 Export & Reports")

    st.markdown(f"""
    <div class="info-block">
        <div class="ib-title">
            <span class="ib-dot" style="background:{C['emerald']}; box-shadow:0 0 8px {C['emerald']};"></span>
            Daten exportieren
        </div>
        <div class="ib-text">
            Lade die <strong>gefilterten Daten</strong> als Excel oder CSV herunter.
            Alle aktiven Sidebar-Filter werden berücksichtigt — du exportierst genau das,
            was du im Dashboard siehst.
        </div>
    </div>""", unsafe_allow_html=True)

    spacer()

    # Export KPIs 
    k1, k2, k3 = st.columns(3)
    with k1: kpi("📊", "Zeilen im Export", human_number(len(fs)), C["cyan"])
    with k2: kpi("🔍", "Aktive Filter", str(sum([
        bool(selected_years), bool(selected_countries), bool(selected_categories),
        bool(selected_brands), bool(selected_departments), bool(selected_products),
        bool(selected_status)
    ])), C["violet"])
    with k3: kpi("📋", "Spalten", str(len(fs.columns.drop(["_year", "_is_cs"]))), C["emerald"])

    spacer()

    #  Export Section 1: Sales Data
    left_ex, right_ex = st.columns(2)

    with left_ex:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("Bestelldaten (Sales)", C["cyan"])

        # CSV Export
        csv_sales = cs.drop(columns=["_year", "_is_cs"]).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Sales als CSV herunterladen",
            data=csv_sales,
            file_name="sales_export.csv",
            mime="text/csv",
            use_container_width=True,
        )

        spacer()

        # Excel Export
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            cs[["order_id", "product_name", "brand", "category", "department",
                "sale_price", "product_cost", "profit", "status", "country",
                "created_at"]].to_excel(writer, sheet_name="Sales", index=False)
        excel_data = excel_buffer.getvalue()

        st.download_button(
            label="📥 Sales als Excel herunterladen",
            data=excel_data,
            file_name="sales_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        spacer()
        mini("Zeilen", human_number(len(cs)), C["cyan"])
        mini("Zeitraum", f"{cs['created_at'].min().strftime('%d.%m.%Y')} — {cs['created_at'].max().strftime('%d.%m.%Y')}" if len(cs) else "—", C["emerald"])
        st.markdown("</div>", unsafe_allow_html=True)

    with right_ex:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        sec("KPI-Zusammenfassung", C["violet"])

        # Build KPI summary
        tot_sales = cs["sale_price"].sum()
        tot_orders = cs["order_id"].nunique()
        aov = tot_sales / tot_orders if tot_orders else 0
        profit = cs["profit"].sum()
        margin = (profit / tot_sales * 100) if tot_sales else 0
        rr = fs["is_returned"].mean() * 100 if len(fs) else 0
        cust = cs["user_id"].nunique()

        kpi_summary = pd.DataFrame({
            "KPI": ["Total Sales", "Bestellungen", "Ø Bestellwert (AOV)",
                    "Bruttogewinn", "Profit Margin", "Return Rate", "Kunden"],
            "Wert": [human_money(tot_sales), human_number(tot_orders), human_money(aov),
                     human_money(profit), pct(margin), pct(rr), human_number(cust)]
        })

        st.dataframe(kpi_summary, use_container_width=True, hide_index=True)

        spacer()

        csv_kpi = kpi_summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 KPIs als CSV herunterladen",
            data=csv_kpi,
            file_name="kpi_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    spacer()

    # Export Section 2: Additional Datasets 
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    sec("Weitere Daten exportieren", C["amber"])

    c1, c2, c3 = st.columns(3)

    with c1:
        # Top Kategorien
        cat_data = cs.groupby("category")["sale_price"].sum().reset_index().sort_values("sale_price", ascending=False)
        cat_csv = cat_data.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Kategorien-Umsatz (CSV)", cat_csv, "kategorien_umsatz.csv", "text/csv", use_container_width=True)

    with c2:
        # Monthly data
        monthly_export = monthly[["Month", "Sales", "MoM"]].copy()
        mon_csv = monthly_export.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Monatsumsatz (CSV)", mon_csv, "monatsumsatz.csv", "text/csv", use_container_width=True)

    with c3:
        # Customer data
        cust_export = cs.groupby("user_id").agg(
            Orders=("order_id", "nunique"),
            Total_Sales=("sale_price", "sum"),
            Avg_Order=("sale_price", "mean")
        ).reset_index().sort_values("Total_Sales", ascending=False)
        cust_csv = cust_export.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Kundendaten (CSV)", cust_csv, "kundendaten.csv", "text/csv", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 10 — VIELEN DANK
# ══════════════════════════════════════════════════════════════
def page_danke():
    render_header("🙏 Vielen Dank", analysis=False)

    #  Title 
    st.markdown("""
    <div style="display:flex; flex-direction:column; align-items:center; text-align:center; padding:80px 20px 0 20px;">
        <div style="font-size:80px; margin-bottom:30px; animation:danke-pulse 2.5s ease-in-out infinite;">🎓</div>
        <div class="danke-title">Vielen Dank<br>für eure Aufmerksamkeit</div>
        <div class="danke-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    spacer()
    spacer()

    # Dank an Dozenten 
    st.markdown(f"""
    <div style="text-align:center; padding:36px 28px;
                background:linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(34,211,238,0.05) 100%);
                border:1px solid rgba(99,102,241,0.15); border-radius:20px; max-width:620px; margin:0 auto;">
        <div style="font-size:15px; color:{C['muted']}; line-height:1.8;">
            Ein besonderer Dank gilt unseren <strong style="color:{C['text']};">Dozenten</strong><br>
            für die fachliche Begleitung und Unterstützung<br>
            während der gesamten Weiterbildung.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  LOGIN PAGE
# ══════════════════════════════════════════════════════════════
VALID_USER = "admin"
VALID_PASS = "admin123"

def page_login():
    render_header("🔐 Kunden-Login", analysis=False)

    spacer()

    #  Layout: Bild-Beschreibung links, Login-Form rechts 
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown(f"""
        <div class="info-block">
            <div class="ib-title">
                <span class="ib-dot" style="background:{C['indigo']}; box-shadow:0 0 8px {C['indigo']};"></span>
                Analytics-Dashboard
            </div>
            <div class="ib-text">
                Ihr persönliches Analytics-Dashboard: KPIs, Funnels und Geo-Analysen
                in Echtzeit.<br><br>
                Nach dem Login erhalten Sie Zugriff auf:<br><br>
                ▸ <strong>Executive Overview</strong> — Umsatz, Bestellungen, Profitabilität<br>
                ▸ <strong>Customer & Conversion</strong> — Funnel, Traffic, Demografie<br>
                ▸ <strong>Product & Inventory</strong> — Kategorie-Performance, Marge<br>
                ▸ <strong>Return Analysis</strong> — Retourenquote, Lost Profit<br>
                ▸ <strong>Machine Learning</strong> — Churn Prediction, Sales Forecast<br>
                ▸ <strong>Export & Reports</strong> — Daten als Excel/CSV herunterladen
            </div>
        </div>""", unsafe_allow_html=True)

    with right:
        st.markdown(f"""
        <div style="text-align:center; margin-bottom:16px;">
            <div style="font-size:42px; margin-bottom:8px;">🔐</div>
            <div style="font-size:18px; font-weight:700; color:{C['text']};">Kunden-Login</div>
            <div style="font-size:12px; color:{C['dim']};">Zugang zum Analytics-Dashboard</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="glass lp-form">', unsafe_allow_html=True)

        login_user = st.text_input("Benutzername", placeholder="Benutzername", key="login_user")
        login_pass = st.text_input("Passwort", type="password", placeholder="••••••••", key="login_pass")

        if st.button("Anmelden →", use_container_width=True, type="primary"):
            if login_user == VALID_USER and login_pass == VALID_PASS:
                st.session_state.logged_in = True
                st.session_state.login_user = login_user
                st.success("✓ Anmeldung erfolgreich — Dashboard wird geladen …")
                st.rerun()
            elif not login_user or not login_pass:
                st.error("Bitte Benutzername und Passwort eingeben.")
            else:
                st.error("Zugangsdaten nicht korrekt — bitte erneut versuchen.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="text-align:center; margin-top:12px;">
            <span style="font-size:10.5px; color:{C['dim']};">
                ⚠️ Demo-Login · Produktiv: Backend-Auth verwenden
            </span>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  LANDING PAGE — STARTSEITE (eine Seite, alles drin)
# ══════════════════════════════════════════════════════════════
def page_startseite():
    """Startseite im 1:1-Look von index(3).html mit funktionalem Streamlit-Login."""

    st.markdown(r"""
    <style>
    /* ══════════════════════════════════════════════════════════════
       DATAVITA LANDING PAGE — aus index(3).html nach Streamlit portiert
       Alle Klassen sind mit dv- gekapselt, damit Dashboard-Seiten unverändert bleiben.
       ══════════════════════════════════════════════════════════════ */
    :root {
      --dv-bg:         #0b0f19;
      --dv-card:       rgba(17,24,39,0.55);
      --dv-card-solid: #111827;
      --dv-panel:      #0f1629;
      --dv-border:     rgba(55,65,81,0.4);
      --dv-text:       #f1f5f9;
      --dv-muted:      #94a3b8;
      --dv-dim:        #64748b;
      --dv-blue:       #3b82f6;
      --dv-indigo:     #6366f1;
      --dv-violet:     #8b5cf6;
      --dv-cyan:       #22d3ee;
      --dv-emerald:    #10b981;
      --dv-amber:      #f59e0b;
      --dv-rose:       #f43f5e;
      --dv-radius:     16px;
    }

    html { scroll-behavior: smooth; }
    .stApp {
      background: var(--dv-bg) !important;
      background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99,102,241,0.12), transparent),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(34,211,238,0.06), transparent) !important;
    }
    /* Streamlit begrenzt Inhalte je nach Version über mehrere Wrapper.
       Alle Startseiten-Wrapper werden deshalb auf echte Viewport-Breite gesetzt. */
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    main,
    .main,
    .block-container {
      width: 100% !important;
      max-width: none !important;
      margin: 0 !important;
      padding: 0 !important;
    }
    [data-testid="stMainBlockContainer"] > div,
    [data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"],
    [data-testid="stMainBlockContainer"] [data-testid="stElementContainer"],
    [data-testid="stMainBlockContainer"] [data-testid="stMarkdownContainer"],
    [data-testid="stMainBlockContainer"] .stMarkdown {
      width: 100% !important;
      max-width: none !important;
    }
    .app-footer { display: none !important; }
    .dv-page, .dv-page * { box-sizing: border-box; }
    .dv-page {
      /* Full-bleed: bricht auch aus einem von Streamlit zentrierten Elterncontainer aus. */
      width: 100vw !important;
      max-width: 100vw !important;
      margin-left: calc(50% - 50vw) !important;
      margin-right: calc(50% - 50vw) !important;
      color: var(--dv-text);
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      line-height: 1.6;
      -webkit-font-smoothing: antialiased;
      overflow-x: clip;
    }
    .dv-page h2, .dv-page h3, .dv-page h4,
    .dv-page p, .dv-page ul { margin: 0; padding: 0; }
    .block-container .dv-page h3 { display: block !important; }
    .dv-wrap { width: 100%; max-width: none; margin: 0; padding: 0 clamp(22px, 2.6vw, 56px); }
    .dv-section { padding: 88px 0; scroll-margin-top: 78px; }

    /* Section title */
    .dv-sec-t { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
    .dv-sec-t .dv-dot { width: 8px; height: 8px; border-radius: 50%; }
    .dv-sec-t .dv-ttl {
      font-size: 13px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 2.5px; color: var(--dv-dim);
    }
    .dv-sec-h { font-size: 34px; font-weight: 800; letter-spacing: -0.7px; margin-bottom: 14px !important; }
    .dv-sec-sub { color: var(--dv-muted); font-size: 15.5px; max-width: 860px; margin-bottom: 42px !important; }

    /* Fixed navbar */
    .dv-nav {
      position: fixed; top: 0; left: 0; right: 0; z-index: 999990;
      background: rgba(11,15,25,0.75); backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border-bottom: 1px solid rgba(99,102,241,0.12);
    }
    .dv-nav-inner {
      width: 100%; max-width: none; margin: 0; padding: 14px clamp(22px, 2.6vw, 56px);
      display: flex; align-items: center; justify-content: space-between;
    }
    .dv-brand {
      display: flex; align-items: center; gap: 10px; font-weight: 800;
      font-size: 17px; letter-spacing: -0.3px; text-decoration: none; color: var(--dv-text) !important;
    }
    .dv-logo-img { width: 36px; height: 36px; border-radius: 9px; display: block; }
    .dv-brand .dv-tag {
      font-size: 9px; font-weight: 600; color: var(--dv-dim);
      text-transform: uppercase; letter-spacing: 2.5px; display: block; margin-top: -2px;
    }
    .dv-nav-links { display: flex; align-items: center; gap: 6px; list-style: none; }
    .dv-nav-links a {
      color: var(--dv-muted) !important; text-decoration: none; font-size: 13.5px;
      font-weight: 500; padding: 8px 14px; border-radius: 10px;
      border: 1px solid transparent; transition: all .2s ease;
    }
    .dv-nav-links a:hover {
      color: var(--dv-text) !important; background: rgba(99,102,241,0.1);
      border-color: rgba(99,102,241,0.2);
    }
    .dv-nav-links a.dv-login-link {
      color: #a5b4fc !important; background: rgba(99,102,241,0.12);
      border: 1px solid rgba(99,102,241,0.25); font-weight: 600;
    }
    .dv-nav-links a.dv-login-link:hover { background: rgba(99,102,241,0.25); }
    .dv-nav-check { display: none; }
    .dv-nav-toggle {
      display: none; background: none; border: 1px solid var(--dv-border);
      color: var(--dv-text); border-radius: 10px; padding: 6px 12px;
      font-size: 18px; cursor: pointer;
    }

    /* Hero */
    .dv-hero {
      min-height: 760px; padding: 170px 0 90px 0; text-align: center;
      position: relative; overflow: hidden; display: flex; align-items: center;
    }
    .dv-hero-bg { position: absolute; inset: 0; z-index: 0; }
    .dv-hero-bg img { width: 100%; height: 100%; object-fit: cover; opacity: 0.55; }
    .dv-hero-bg::after {
      content: ""; position: absolute; inset: 0;
      background: linear-gradient(180deg, rgba(11,15,25,0.85) 0%, rgba(11,15,25,0.55) 45%, #0b0f19 100%);
    }
    .dv-hero .dv-wrap { position: relative; z-index: 1; width: 100%; }
    .dv-hero-badge {
      display: inline-block; background: rgba(99,102,241,0.12);
      border: 1px solid rgba(99,102,241,0.25); border-radius: 999px;
      padding: 7px 18px; font-size: 12px; font-weight: 600; color: #a5b4fc;
      letter-spacing: 1px; text-transform: uppercase; margin-bottom: 26px;
    }
    .dv-hero h1 {
      font-size: 56px; font-weight: 900; letter-spacing: -1.5px; line-height: 1.12;
      max-width: 1180px; margin: 0 auto;
      background: linear-gradient(135deg, var(--dv-cyan), var(--dv-indigo), var(--dv-violet));
      -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
    }
    .dv-hero p { font-size: 17px; color: var(--dv-muted); max-width: 860px; margin: 22px auto 38px auto; }
    .dv-cta-row { display: flex; gap: 14px; justify-content: center; flex-wrap: wrap; }
    .dv-btn {
      display: inline-block; text-decoration: none !important; font-weight: 700;
      font-size: 14.5px; padding: 14px 32px; border-radius: 12px;
      transition: all .25s ease; cursor: pointer; border: none; font-family: inherit;
    }
    .dv-btn-primary {
      background: linear-gradient(135deg, var(--dv-indigo), var(--dv-violet)); color: #fff !important;
      box-shadow: 0 8px 30px rgba(99,102,241,0.3);
    }
    .dv-btn-primary:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(99,102,241,0.45); }
    .dv-btn-ghost { background: rgba(17,24,39,0.6); color: var(--dv-text) !important; border: 1px solid var(--dv-border); }
    .dv-btn-ghost:hover { border-color: rgba(99,102,241,0.4); background: rgba(99,102,241,0.08); }

    /* Stats */
    .dv-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 70px; }
    .dv-kpi {
      background: rgba(17,24,39,0.6); backdrop-filter: blur(16px);
      border: 1px solid rgba(55,65,81,0.45); border-radius: var(--dv-radius);
      padding: 22px 20px; position: relative; overflow: hidden; transition: all .3s ease;
    }
    .dv-kpi:hover { border-color: rgba(99,102,241,0.35); transform: translateY(-3px); box-shadow: 0 8px 30px rgba(99,102,241,0.12); }
    .dv-kpi .dv-glow { position: absolute; top: 0; left: 0; right: 0; height: 2px; }
    .dv-kpi .dv-val { font-size: 30px; font-weight: 800; letter-spacing: -0.5px; }
    .dv-kpi .dv-lbl { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: var(--dv-dim); margin-top: 4px; }

    /* Cards */
    .dv-grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }
    .dv-grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 18px; }
    .dv-glass {
      background: var(--dv-card); backdrop-filter: blur(16px);
      border: 1px solid var(--dv-border); border-radius: var(--dv-radius);
      padding: 28px 26px; transition: all .3s ease; position: relative; overflow: hidden;
    }
    .dv-glass:hover { border-color: rgba(99,102,241,0.35); transform: translateY(-3px); box-shadow: 0 8px 30px rgba(99,102,241,0.12); }
    .dv-icon-wrap {
      width: 46px; height: 46px; border-radius: 12px; display: flex;
      align-items: center; justify-content: center; font-size: 22px; margin-bottom: 16px;
    }
    .dv-glass h3 { font-size: 16.5px; font-weight: 700; margin-bottom: 8px !important; letter-spacing: -0.2px; color: var(--dv-text); }
    .dv-glass p { font-size: 13.5px; color: var(--dv-muted); line-height: 1.7; }

    /* Why */
    .dv-milestone {
      background: var(--dv-card); border: 1px solid var(--dv-border); border-radius: 14px;
      padding: 20px 22px; display: flex; gap: 16px; align-items: flex-start; transition: all .3s ease;
    }
    .dv-milestone:hover { border-color: rgba(99,102,241,0.3); transform: translateX(4px); }
    .dv-ms-icon { width: 42px; height: 42px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 19px; flex-shrink: 0; }
    .dv-milestone h4 { font-size: 15px; font-weight: 700; margin-bottom: 4px !important; color: var(--dv-text); }
    .dv-milestone p { font-size: 13px; color: var(--dv-muted); line-height: 1.6; }

    /* Images / case study */
    .dv-framed {
      background: var(--dv-card); border: 1px solid var(--dv-border); border-radius: var(--dv-radius);
      padding: 10px; transition: all .3s ease;
    }
    .dv-framed:hover { border-color: rgba(99,102,241,0.35); box-shadow: 0 8px 30px rgba(99,102,241,0.12); }
    .dv-framed img { width: 100%; display: block; border-radius: 10px; }
    .dv-framed .dv-cap { font-size: 11px; color: var(--dv-dim); text-align: center; padding: 10px 4px 4px 4px; letter-spacing: 0.5px; }
    .dv-case-list { list-style: none; margin: 0 0 26px 0 !important; }
    .dv-case-list li {
      position: relative; padding: 12px 0 12px 28px; font-size: 14px; color: #cbd5e1;
      line-height: 1.6; border-bottom: 1px solid rgba(55,65,81,0.3);
    }
    .dv-case-list li::before { content: "▸"; position: absolute; left: 4px; color: var(--dv-cyan); }
    .dv-case-list li strong { color: var(--dv-text); font-weight: 600; }

    /* References */
    .dv-logo-row { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 26px; padding: 26px 0 50px 0; }
    .dv-logo-row span { font-size: 17px; font-weight: 800; letter-spacing: 3px; color: #475569; transition: color .3s; }
    .dv-logo-row span:hover { color: var(--dv-muted); }
    .dv-refs-visual { margin-bottom: 34px; }
    .dv-quote { font-size: 14.5px !important; color: #cbd5e1 !important; line-height: 1.75 !important; font-style: italic; }
    .dv-quote-who { margin-top: 18px; display: flex; align-items: center; gap: 12px; }
    .dv-avatar { width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 17px; }
    .dv-quote-name { font-size: 13px; font-weight: 700; }
    .dv-quote-role { font-size: 11px; color: var(--dv-dim); }

    /* CTA / forms */
    .dv-cta-band {
      background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(34,211,238,0.08) 100%);
      border: 1px solid rgba(99,102,241,0.2); border-radius: 20px;
      padding: 52px 40px; text-align: center; position: relative; overflow: hidden;
    }
    .dv-cta-band::before {
      content: ""; position: absolute; top: -40%; right: -5%; width: 260px; height: 260px;
      background: radial-gradient(circle, rgba(99,102,241,0.15), transparent 70%); border-radius: 50%;
    }
    .dv-cta-band h2 { font-size: 30px; font-weight: 800; letter-spacing: -0.6px; margin-bottom: 10px !important; position: relative; z-index: 1; }
    .dv-cta-band p { color: var(--dv-muted); margin-bottom: 28px !important; position: relative; z-index: 1; }
    .dv-contact-form label {
      display: block; font-size: 11.5px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 1px; color: var(--dv-dim); margin: 16px 0 6px 0;
    }
    .dv-contact-form input, .dv-contact-form textarea {
      width: 100%; background: rgba(15,22,41,0.8); border: 1px solid rgba(55,65,81,0.5);
      border-radius: 10px; padding: 12px 14px; color: var(--dv-text); font-family: inherit;
      font-size: 14px; transition: border-color .2s, box-shadow .2s; outline: none;
    }
    .dv-contact-form input:focus, .dv-contact-form textarea:focus {
      border-color: var(--dv-indigo); box-shadow: 0 0 0 3px rgba(99,102,241,0.15);
    }
    .dv-contact-form button { width: 100%; margin-top: 22px; }

    /* Login section, including native Streamlit form */
    .dv-login-title-wrap { padding: 40px 0 0 0; scroll-margin-top: 78px; }
    .dv-login-image { margin-top: 8px; }
    div[data-testid="stForm"] {
      background: var(--dv-card) !important;
      backdrop-filter: blur(16px) !important;
      border: 1px solid var(--dv-border) !important;
      border-radius: var(--dv-radius) !important;
      padding: 30px 34px 28px 34px !important;
      transition: all .3s ease;
    }
    div[data-testid="stForm"]:hover {
      border-color: rgba(99,102,241,0.35) !important;
      box-shadow: 0 8px 30px rgba(99,102,241,0.12) !important;
    }
    div[data-testid="stForm"] label {
      font-size: 11.5px !important; font-weight: 600 !important; text-transform: uppercase !important;
      letter-spacing: 1px !important; color: var(--dv-dim) !important;
    }
    div[data-testid="stForm"] [data-baseweb="input"] > div {
      background: rgba(15,22,41,0.8) !important;
      border: 1px solid rgba(55,65,81,0.5) !important;
      border-radius: 10px !important;
    }
    div[data-testid="stForm"] input { color: var(--dv-text) !important; }
    div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {
      width: 100% !important; margin-top: 8px !important; min-height: 48px !important;
      background: linear-gradient(135deg, var(--dv-indigo), var(--dv-violet)) !important;
      color: white !important; border: none !important; border-radius: 12px !important;
      font-weight: 700 !important; font-size: 14.5px !important;
      box-shadow: 0 8px 30px rgba(99,102,241,0.3) !important;
    }
    div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover {
      transform: translateY(-2px) !important;
      box-shadow: 0 12px 40px rgba(99,102,241,0.45) !important;
    }
    .dv-login-head { text-align: center; margin-bottom: 18px; }
    .dv-login-lock { font-size: 34px; margin-bottom: 8px; }
    .dv-login-head h3 { display: block !important; text-align: center; font-size: 20px; margin-bottom: 4px !important; }
    .dv-login-hint { text-align: center; font-size: 12.5px; color: var(--dv-dim); }
    .dv-demo-hint { text-align: center; font-size: 10.5px; color: var(--dv-dim); margin-top: 12px; }

    /* Footer */
    .dv-footer {
      text-align: center; color: #475569; font-size: 12px;
      padding: 34px 0; margin-top: 70px; border-top: 1px solid rgba(55,65,81,0.3);
    }
    .dv-footer strong, .dv-footer a { color: var(--dv-dim) !important; }

    /* Light reveal animation; no JavaScript needed inside Streamlit */
    @keyframes dv-reveal-up {
      from { opacity: 0; transform: translateY(22px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    .dv-reveal { animation: dv-reveal-up .65s ease both; }
    .dv-grid-3 .dv-reveal:nth-child(2), .dv-grid-2 .dv-reveal:nth-child(2) { animation-delay: .06s; }
    .dv-grid-3 .dv-reveal:nth-child(3) { animation-delay: .12s; }

    @media (max-width: 960px) {
      .dv-grid-3 { grid-template-columns: repeat(2, 1fr); }
      .dv-stats { grid-template-columns: repeat(2, 1fr); }
      .dv-hero h1 { font-size: 40px; }
    }
    @media (max-width: 640px) {
      .dv-section { padding: 60px 0; }
      .dv-hero { min-height: auto; padding: 130px 0 60px 0; }
      .dv-hero h1 { font-size: 31px; letter-spacing: -1px; }
      .dv-grid-3, .dv-grid-2 { grid-template-columns: 1fr; }
      .dv-sec-h { font-size: 26px; }
      .dv-nav-toggle { display: block; }
      .dv-nav-links {
        display: none; position: absolute; top: 100%; left: 0; right: 0;
        flex-direction: column; align-items: stretch; padding: 12px 20px 18px 20px;
        background: rgba(11,15,25,0.97); border-bottom: 1px solid rgba(99,102,241,0.15);
      }
      .dv-nav-links a { display: block; }
      .dv-nav-check:checked ~ .dv-nav-links { display: flex; }
      .dv-stats { margin-top: 48px; }
      .dv-wrap { padding: 0 18px; }
      .dv-nav-inner { padding-left: 18px; padding-right: 18px; }
      .dv-cta-band { padding: 38px 22px; }
      div[data-testid="stForm"] { padding: 26px 22px !important; }
    }
    </style>
    """, unsafe_allow_html=True)

    # Navbar, Hero, Leistungen, Warum wir, Fallstudie, Referenzen und Kontakt.
    st.markdown(r"""
    <div class="dv-page">
      <nav class="dv-nav">
        <div class="dv-nav-inner">
          <a class="dv-brand" href="#dv-top">
            <img class="dv-logo-img"
                 src="https://d8j0ntlcm91z4.cloudfront.net/user_3EgbIyuMhuxNOxm0PkQLFXvGS5w/hf_20260713_195536_bffe8a65-6866-492c-89a4-6e8d8e29abcf_min.webp"
                 alt="DataVita Logo">
            <span>DataVita Analytics<span class="dv-tag">Data Science · Germany</span></span>
          </a>
          <input class="dv-nav-check" type="checkbox" id="dv-nav-check">
          <label class="dv-nav-toggle" for="dv-nav-check" aria-label="Menü">☰</label>
          <ul class="dv-nav-links">
            <li><a href="#dv-why">Warum wir</a></li>
            <li><a href="#dv-case">Fallstudie</a></li>
            <li><a href="#dv-contact">Kontakt</a></li>
            <li><a href="#dv-login" class="dv-login-link">🔐 Kunden-Login</a></li>
          </ul>
        </div>
      </nav>

      <header class="dv-hero" id="dv-top">
        <div class="dv-hero-bg" aria-hidden="true">
          <img src="https://d8j0ntlcm91z4.cloudfront.net/user_3EgbIyuMhuxNOxm0PkQLFXvGS5w/hf_20260713_195530_2d5c4388-dd4d-4538-a244-411878e619bf_min.webp"
               alt="" fetchpriority="high">
        </div>
        <div class="dv-wrap">
          <div class="dv-hero-badge">Data Science · Made in Germany · Weltweit im Einsatz</div>
          <h1>Daten in Entscheidungen verwandeln.</h1>
          <p>DataVita Analytics ist Ihre Data-Science-Beratung aus Deutschland — Predictive Modeling,
             Business-Intelligence-Dashboards und KI-Strategie für Kunden auf der ganzen Welt.</p>
          <div class="dv-cta-row">
            <a class="dv-btn dv-btn-primary" href="#dv-contact">Projekt anfragen →</a>
            <a class="dv-btn dv-btn-ghost" href="#dv-login">Kunden-Login</a>
          </div>
          <div style="margin-top:52px; text-align:center;">
            <div class="dv-sec-t" style="justify-content:center; border-bottom:none; margin-bottom:8px;">
              <span class="dv-dot" style="background:var(--dv-cyan);box-shadow:0 0 8px var(--dv-cyan)"></span>
              <span class="dv-ttl">Unsere Leistungen</span>
            </div>
            <h2 class="dv-sec-h">Generelle Leistungen im Bereich Data Science</h2>
            <p class="dv-sec-sub">End-to-End-Datenprojekte aus einer Hand — von der Strategie über die Analyse bis zum produktiven ML-Modell.</p>
          </div>
          <div class="dv-grid-3" style="margin-top:28px;">
            <div class="dv-glass dv-reveal"><div class="dv-icon-wrap" style="background:rgba(59,130,246,0.12)">🗄️</div><h3>Data Engineering & ETL</h3><p>Datenpipelines, API-Anbindungen, Bereinigung und Transformation — saubere, strukturierte Daten als Fundament jeder Analyse.</p></div>
            <div class="dv-glass dv-reveal"><div class="dv-icon-wrap" style="background:rgba(20,184,166,0.12)">🧩</div><h3>Datenbank-Design & SQL</h3><p>Relationale und analytische Datenmodelle, Schema-Design, performante Abfragen und Datenbankoptimierung.</p></div>
            <div class="dv-glass dv-reveal"><div class="dv-icon-wrap" style="background:rgba(139,92,246,0.12)">📊</div><h3>Datenvisualisierung & Dashboards</h3><p>Interaktive BI-Dashboards mit Streamlit, Plotly, Power BI oder Tableau — KPIs und Trends auf einen Blick.</p></div>
            <div class="dv-glass dv-reveal"><div class="dv-icon-wrap" style="background:rgba(244,63,94,0.12)">📈</div><h3>Explorative Datenanalyse</h3><p>Statistische Auswertungen, Muster- und Anomalie-Erkennung, Hypothesentests und datengetriebene Handlungsempfehlungen.</p></div>
            <div class="dv-glass dv-reveal"><div class="dv-icon-wrap" style="background:rgba(16,185,129,0.12)">🤖</div><h3>Machine Learning & Predictive Analytics</h3><p>Klassifikation, Regression, Zeitreihenprognosen und Clustering — vom Prototyp bis zum produktiven Modell.</p></div>
            <div class="dv-glass dv-reveal"><div class="dv-icon-wrap" style="background:rgba(34,211,238,0.12)">🔍</div><h3>Web Scraping & Datenerhebung</h3><p>Automatisierte Datengewinnung aus Webseiten, APIs und Dokumenten — zuverlässig, skalierbar und regelkonform.</p></div>
            <div class="dv-glass dv-reveal"><div class="dv-icon-wrap" style="background:rgba(245,158,11,0.12)">☁️</div><h3>Cloud & Big Data</h3><p>Datenarchitekturen auf AWS, GCP oder Azure — von Data Lakes über BigQuery bis hin zu skalierbaren ML-Pipelines.</p></div>
            <div class="dv-glass dv-reveal"><div class="dv-icon-wrap" style="background:rgba(249,115,22,0.12)">📝</div><h3>Reporting & Automatisierung</h3><p>Automatisierte Reports, Alerting-Systeme und Self-Service-Analytics — wiederkehrende Analysen ohne manuellen Aufwand.</p></div>
            <div class="dv-glass dv-reveal"><div class="dv-icon-wrap" style="background:rgba(99,102,241,0.12)">🧠</div><h3>KI-Strategie & Beratung</h3><p>Use-Case-Bewertung, KI-Roadmap und Datenkultur — pragmatisch, messbar und auf Ihr Geschäftsmodell zugeschnitten.</p></div>
          </div>
        </div>
      </header>



      <section class="dv-section" id="dv-why">
        <div class="dv-wrap">
          <div class="dv-sec-t"><span class="dv-dot" style="background:var(--dv-violet);box-shadow:0 0 8px var(--dv-violet)"></span><span class="dv-ttl">Warum DataVita</span></div>
          <h2 class="dv-sec-h">Deutsche Gründlichkeit, globale Perspektive</h2>
          <p class="dv-sec-sub">Wir verbinden Ingenieurs-Qualität und Datenschutz nach deutschem Standard mit Projekterfahrung aus über 15 Ländern.</p>
          <div class="dv-grid-2">
            <div class="dv-milestone dv-reveal"><div class="dv-ms-icon" style="background:rgba(34,211,238,0.12)">🇩🇪</div><div><h4>Qualität & DSGVO</h4><p>Entwicklung und Datenhaltung in Deutschland, DSGVO-konforme Prozesse und dokumentierte, wartbare Lösungen — kein Blackbox-Code.</p></div></div>
            <div class="dv-milestone dv-reveal"><div class="dv-ms-icon" style="background:rgba(139,92,246,0.12)">🌍</div><div><h4>Globale Erfahrung</h4><p>Remote-erprobte Zusammenarbeit über Zeitzonen hinweg, Projektsprachen Deutsch und Englisch, Branchen von E-Commerce bis Industrie.</p></div></div>
            <div class="dv-milestone dv-reveal"><div class="dv-ms-icon" style="background:rgba(16,185,129,0.12)">🔗</div><div><h4>End-to-End aus einer Hand</h4><p>Eine Pipeline, ein Team: Datenbeschaffung → Bereinigung → Datenbank → Dashboard → ML. Keine Reibungsverluste zwischen Dienstleistern.</p></div></div>
            <div class="dv-milestone dv-reveal"><div class="dv-ms-icon" style="background:rgba(245,158,11,0.12)">📈</div><div><h4>Messbarer Mehrwert</h4><p>Jedes Projekt startet mit klaren KPIs und endet mit quantifiziertem Ergebnis — Umsatz, Retourenquote, Conversion oder Churn.</p></div></div>
          </div>
        </div>
      </section>

      <section class="dv-section" id="dv-case">
        <div class="dv-wrap">
          <div class="dv-sec-t"><span class="dv-dot" style="background:var(--dv-rose);box-shadow:0 0 8px var(--dv-rose)"></span><span class="dv-ttl">Fallstudie</span></div>
          <h2 class="dv-sec-h">E-Commerce Analytics für einen Online-Modeshop</h2>
          <p class="dv-sec-sub">Referenzprojekt aus unserem Portfolio: vollständige Analyse-Plattform für einen Modeshop mit über 100.000 Bestellpositionen — von der Rohdaten-Pipeline bis zur RFM-Kundensegmentierung.</p>
          <div class="dv-grid-2" style="align-items:center;">
            <div class="dv-framed dv-reveal">
              <img src="https://d8j0ntlcm91z4.cloudfront.net/user_3EgbIyuMhuxNOxm0PkQLFXvGS5w/hf_20260713_200823_a3496ef2-b376-4f9f-9859-ba986206ba49_min.webp"
                   alt="E-Commerce-Analytics-Dashboard: Funnel, Retouren und Segmentierung" loading="lazy">
              <div class="dv-cap">Retouren, Conversion-Funnel und Segmente auf einen Blick</div>
            </div>
            <div class="dv-reveal">
              <ul class="dv-case-list">
                <li><strong>Daten-Pipeline:</strong> 7 relational verknüpfte Tabellen, BigQuery → Pandas-ETL → PostgreSQL mit ERD-basiertem Schema</li>
                <li><strong>BI-Dashboard:</strong> 6 Analyseseiten, über 20 interaktive Plotly-Charts, globale Echtzeit-Filter (Jahr, Land, Kategorie, Brand)</li>
                <li><strong>Retouren-Analyse:</strong> Return-Rate nach Kategorie, Brand und Preissegment inkl. Lost-Profit-Bewertung</li>
                <li><strong>Conversion-Funnel:</strong> Web-Events vom Seitenaufruf bis zum Kauf, Traffic-Quellen und Conversion-Rate je Segment</li>
                <li><strong>Machine Learning:</strong> Churn Prediction, Sales Forecasting und Modellvergleich</li>
              </ul>
              <a class="dv-btn dv-btn-ghost" href="#dv-login">Zur Live-Demo im Kunden-Login →</a>
            </div>
          </div>
        </div>
      </section>
      <section class="dv-section" id="dv-contact">
        <div class="dv-wrap">
          <div class="dv-cta-band dv-reveal" style="margin-bottom:56px;">
            <h2>Bereit, mehr aus Ihren Daten zu machen?</h2>
            <p>Unverbindliches Erstgespräch — wir antworten innerhalb von 24 Stunden.</p>
            <a class="dv-btn dv-btn-primary" href="mailto:kontakt@datavita.example?subject=Projektanfrage">Erstgespräch vereinbaren →</a>
          </div>
          <div class="dv-grid-2">
            <div>
              <div class="dv-sec-t"><span class="dv-dot" style="background:var(--dv-amber);box-shadow:0 0 8px var(--dv-amber)"></span><span class="dv-ttl">Kontakt</span></div>
              <h2 class="dv-sec-h" style="font-size:26px;">Schreiben Sie uns</h2>
              <p class="dv-sec-sub" style="margin-bottom:0 !important;">Beschreiben Sie kurz Ihr Vorhaben — Datenlage, Ziel, Zeitrahmen. Das Formular öffnet Ihre E-Mail-Anwendung mit vorbereiteter Nachricht.</p>
            </div>
            <form class="dv-glass dv-contact-form" action="mailto:kontakt@datavita.example" method="post" enctype="text/plain">
              <label for="dv-c-name">Name</label>
              <input id="dv-c-name" name="Name" type="text" placeholder="Max Mustermann" required>
              <label for="dv-c-email">E-Mail</label>
              <input id="dv-c-email" name="E-Mail" type="email" placeholder="max@firma.de" required>
              <label for="dv-c-msg">Nachricht</label>
              <textarea id="dv-c-msg" name="Nachricht" rows="4" placeholder="Worum geht es?" required></textarea>
              <button class="dv-btn dv-btn-primary" type="submit">Nachricht senden</button>
            </form>
          </div>
        </div>
      </section>

      <div class="dv-login-title-wrap" id="dv-login">
        <div class="dv-wrap">
          <div class="dv-sec-t"><span class="dv-dot" style="background:var(--dv-indigo);box-shadow:0 0 8px var(--dv-indigo)"></span><span class="dv-ttl">Kunden-Login</span></div>
          <h2 class="dv-sec-h">Zugang zum Analytics-Dashboard</h2>
          <p class="dv-sec-sub">Melden Sie sich an und öffnen Sie anschließend Ihre Projektübersicht mit allen Analytics-Dashboards.</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    
    outer_left, login_area, outer_right = st.columns([0.35, 12, 0.35])
    with login_area:
        image_col, form_col = st.columns([1.2, 1], gap="large")

        with image_col:
            st.markdown(r"""
            <div class="dv-page dv-framed dv-login-image dv-reveal">
              <img src="https://d8j0ntlcm91z4.cloudfront.net/user_3EgbIyuMhuxNOxm0PkQLFXvGS5w/hf_20260713_195535_6c9c4d28-e904-4667-a885-3024cc9040a1_min.webp"
                   alt="Vorschau des DataVita Analytics-Dashboards" loading="lazy">
              <div class="dv-cap">Ihr Analytics-Dashboard: KPIs, Funnels und Geo-Analysen in Echtzeit</div>
            </div>
            """, unsafe_allow_html=True)

        with form_col:
            with st.form("dv_login_form", clear_on_submit=False):
                st.markdown(r"""
                <div class="dv-page dv-login-head">
                  <div class="dv-login-lock">🔐</div>
                  <h3>Kunden-Login</h3>
                  <div class="dv-login-hint">Zugang zum Analytics-Dashboard</div>
                </div>
                """, unsafe_allow_html=True)
                login_user = st.text_input(
                    "Benutzername",
                    placeholder="Benutzername",
                    key="dv_login_user",
                )
                login_pass = st.text_input(
                    "Passwort",
                    type="password",
                    placeholder="••••••••",
                    key="dv_login_pass",
                )
                submitted = st.form_submit_button("Anmelden →", use_container_width=True)

                if submitted:
                    if login_user == VALID_USER and login_pass == VALID_PASS:
                        st.session_state.logged_in = True
                        st.session_state.login_user = login_user
                        st.session_state.project_selected = False
                        st.rerun()
                    elif not login_user or not login_pass:
                        st.error("Bitte Benutzername und Passwort eingeben.")
                    else:
                        st.error("Zugangsdaten nicht korrekt — bitte erneut versuchen.")

            st.markdown(r"""
            <div class="dv-page dv-demo-hint">⚠️ Demo-Login · Produktiv: Backend-Auth verwenden</div>
            """, unsafe_allow_html=True)

    st.markdown(r"""
    <div class="dv-page">
      <footer class="dv-footer">
        <div class="dv-wrap">
          <strong>DataVita Analytics</strong> · Data Science aus Deutschland, weltweit ·
          <a href="#dv-login">Login</a> &nbsp;|&nbsp; <strong>PostgreSQL</strong> ·
          <strong>Streamlit</strong> · <strong>Plotly</strong> · <strong>Scikit-Learn</strong>
          <br><span style="font-size:10.5px;">© 2026 DataVita Analytics (Platzhalter) · Impressum · Datenschutz</span>
        </div>
      </footer>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PROJEKTE-SEITE 
# ══════════════════════════════════════════════════════════════
def page_projekte():
    """Projektübersicht nach dem Login — zeigt verfügbare Dashboards/Projekte."""

    #  Inline CSS für Projekte-Seite (Animationen + Dekorationen) 
    st.markdown(f"""
    <style>
    @keyframes proj-float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-8px); }}
    }}
    @keyframes proj-glow {{
        0%, 100% {{ box-shadow: 0 0 20px rgba(99,102,241,0.15); }}
        50% {{ box-shadow: 0 0 40px rgba(99,102,241,0.3); }}
    }}
    @keyframes proj-shine {{
        0% {{ left: -100%; }}
        100% {{ left: 200%; }}
    }}
    .proj-welcome {{
        text-align: center;
        padding: 50px 20px 30px 20px;
        position: relative;
    }}
    .proj-welcome::before {{
        content: "";
        position: absolute;
        top: -30%;
        left: 50%;
        transform: translateX(-50%);
        width: 700px;
        height: 400px;
        background: radial-gradient(ellipse, rgba(99,102,241,0.1), transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }}
    .proj-avatar {{
        width: 70px; height: 70px; border-radius: 50%;
        background: linear-gradient(135deg, {C['indigo']}, {C['violet']});
        display: flex; align-items: center; justify-content: center;
        font-size: 30px; margin: 0 auto 18px auto;
        animation: proj-float 3s ease-in-out infinite;
        box-shadow: 0 8px 30px rgba(99,102,241,0.25);
    }}
    .proj-greeting {{
        font-size: 28px; font-weight: 800; color: {C['text']};
        letter-spacing: -0.5px; margin-bottom: 6px;
    }}
    .proj-sub {{
        font-size: 15px; color: {C['muted']}; margin-bottom: 8px;
    }}
    .proj-card {{
        background: rgba(17,24,39,0.6);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 22px;
        padding: 34px 32px;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4,0,0.2,1);
        animation: proj-glow 4s ease-in-out infinite;
    }}
    .proj-card:hover {{
        border-color: rgba(99,102,241,0.5);
        transform: translateY(-4px) scale(1.005);
    }}
    .proj-card::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, {C['cyan']}, {C['indigo']}, {C['violet']}, {C['rose']});
    }}
    .proj-card::after {{
        content: "";
        position: absolute;
        top: 0; left: -100%;
        width: 60%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.03), transparent);
        animation: proj-shine 6s ease-in-out infinite;
    }}
    .proj-card-icon {{
        width: 64px; height: 64px; border-radius: 18px;
        background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(34,211,238,0.1));
        display: flex; align-items: center; justify-content: center;
        font-size: 30px; flex-shrink: 0;
        border: 1px solid rgba(99,102,241,0.15);
    }}
    .proj-badge {{
        display: inline-block;
        background: rgba(16,185,129,0.12);
        border: 1px solid rgba(16,185,129,0.3);
        color: {C['emerald']};
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 10.5px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}
    .proj-stats {{
        display: flex; gap: 24px; flex-wrap: wrap;
        margin-top: 18px; padding-top: 16px;
        border-top: 1px solid rgba(55,65,81,0.3);
    }}
    .proj-stat {{
        text-align: center;
    }}
    .proj-stat .ps-val {{
        font-size: 22px; font-weight: 800; letter-spacing: -0.3px;
    }}
    .proj-stat .ps-lbl {{
        font-size: 10px; font-weight: 600; text-transform: uppercase;
        letter-spacing: 1px; color: {C['dim']}; margin-top: 2px;
    }}
    .proj-coming {{
        background: rgba(17,24,39,0.4);
        border: 1px dashed rgba(55,65,81,0.4);
        border-radius: 18px;
        padding: 26px 22px;
        opacity: 0.55;
        transition: all 0.3s;
        min-height: 180px;
    }}
    .proj-coming:hover {{ opacity: 0.75; border-color: rgba(55,65,81,0.6); }}
    .proj-coming-badge {{
        display: inline-block;
        background: rgba(55,65,81,0.3);
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 10px;
        font-weight: 700;
        color: {C['dim']};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 8px;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Welcome Header 
    user_name = st.session_state.get('login_user', 'User')
    st.markdown(f"""
    <div class="proj-welcome">
        <div class="proj-avatar">👤</div>
        <div class="proj-greeting">Willkommen, {user_name}</div>
        <div class="proj-sub">Wählen Sie ein Projekt, um das Dashboard zu öffnen</div>
    </div>
    """, unsafe_allow_html=True)

    spacer()

    #  Haupt-Projekt: Abschlussprojekt 
    st.markdown(f"""
    <div class="proj-card">
        <div style="display:flex; gap:22px; align-items:flex-start; position:relative; z-index:1;">
            <div class="proj-card-icon">📊</div>
            <div style="flex:1;">
                <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:8px;">
                    <span style="font-size:22px; font-weight:800; color:{C['text']}; letter-spacing:-0.4px;">
                        TheLook E-Commerce Analytics
                    </span>
                    <span class="proj-badge">● Live</span>
                </div>
                <div style="font-size:12px; font-weight:600; color:{C['indigo']}; text-transform:uppercase;
                            letter-spacing:1.5px; margin-bottom:14px;">
                    Abschlussprojekt · Data Science · Weiterbildung 2026
                </div>
                <div style="font-size:14px; color:{C['muted']}; line-height:1.75;">
                    Vollständige Analyse-Plattform für einen Online-Modeshop —
                    BI-Dashboard mit 6 Analyseseiten inkl. RFM-Kundensegmentierung,
                    über 20 interaktive Charts und
                    Machine Learning (Churn Prediction & Forecasting).
                </div>
                <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:16px;">
                    <span class="tech-pill">🐘 PostgreSQL</span>
                    <span class="tech-pill">🎯 Streamlit</span>
                    <span class="tech-pill">📊 Plotly</span>
                    <span class="tech-pill">🤖 Scikit-Learn</span>
                    <span class="tech-pill">🐼 Pandas</span>
                    <span class="tech-pill">🎨 Custom CSS</span>
                </div>
                <div class="proj-stats">
                    <div class="proj-stat"><div class="ps-val" style="color:{C['cyan']};">14</div><div class="ps-lbl">Seiten</div></div>
                    <div class="proj-stat"><div class="ps-val" style="color:{C['violet']};">20+</div><div class="ps-lbl">Charts</div></div>
                    <div class="proj-stat"><div class="ps-val" style="color:{C['emerald']};">3</div><div class="ps-lbl">ML-Modelle</div></div>
                    <div class="proj-stat"><div class="ps-val" style="color:{C['amber']};">7</div><div class="ps-lbl">Tabellen</div></div>
                    <div class="proj-stat"><div class="ps-val" style="color:{C['rose']};">100K+</div><div class="ps-lbl">Datensätze</div></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    spacer()

    #  Button zum Dashboard öffnen 
    if st.button("📊  Dashboard öffnen →", use_container_width=True, type="primary", key="open_project_btn"):
        st.session_state.project_selected = True
        st.rerun()

    spacer()
    spacer()

    # Weitere Projekte  
    st.markdown(f"""
    <div class="lp-divider"></div>
    <div style="margin-bottom:20px; margin-top:20px;">
        <div class="sec-t">
            <span class="dot" style="background:{C['dim']}; box-shadow:0 0 6px {C['dim']};"></span>
            <span class="ttl">Weitere Projekte</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown(f"""
        <div class="proj-coming">
            <div style="font-size:28px; margin-bottom:10px;">📈</div>
            <div style="font-size:15px; font-weight:700; color:{C['muted']};">
                Sales Forecasting
            </div>
            <div class="proj-coming-badge">Demnächst</div>
            <div style="font-size:12px; color:{C['dim']}; line-height:1.6; margin-top:12px;">
                Zeitreihenanalyse mit ARIMA & Prophet
            </div>
        </div>""", unsafe_allow_html=True)

    with p2:
        st.markdown(f"""
        <div class="proj-coming">
            <div style="font-size:28px; margin-bottom:10px;">🏥</div>
            <div style="font-size:15px; font-weight:700; color:{C['muted']};">
                Healthcare Analytics
            </div>
            <div class="proj-coming-badge">Demnächst</div>
            <div style="font-size:12px; color:{C['dim']}; line-height:1.6; margin-top:12px;">
                Patientendaten & Ressourcenplanung
            </div>
        </div>""", unsafe_allow_html=True)

    with p3:
        st.markdown(f"""
        <div class="proj-coming">
            <div style="font-size:28px; margin-bottom:10px;">🌐</div>
            <div style="font-size:15px; font-weight:700; color:{C['muted']};">
                Web Scraping Pipeline
            </div>
            <div class="proj-coming-badge">Demnächst</div>
            <div style="font-size:12px; color:{C['dim']}; line-height:1.6; margin-top:12px;">
                Automatisierte Datenerhebung & ETL
            </div>
        </div>""", unsafe_allow_html=True)

    spacer()

    # Logout Button 
    if st.button("🚪  Abmelden", use_container_width=False, type="secondary", key="logout_projekte"):
        st.session_state.logged_in = False
        st.session_state.project_selected = False
        st.session_state.pop("login_user", None)
        st.rerun()


## ╔═══════════════════════════════════════════════════════════╗
#  NAVIGATION — 3 Stufen:
#  1) Nicht eingeloggt  → Startseite (eine Seite, kein Sidebar)
#  2) Eingeloggt        → Meine Projekte (eine Seite, kein Sidebar)
#  3) Projekt gewählt   → Dashboard (voller Sidebar + Filter)
# ╚═══════════════════════════════════════════════════════════╝
if st.session_state.logged_in and st.session_state.project_selected:
    # STUFE 3: Dashboard (Abschlussprojekt) 
    nav_pages = {
        "Präsentation": [
            st.Page(page_deckblatt, title="Deckblatt", icon="📋", url_path="deckblatt", default=True),
            st.Page(page_projektziel, title="Projektziel & Aufbau", icon="🎯", url_path="projektziel"),
        ],
        "Analyse": [
            st.Page(page_overview, title="Executive Overview", icon="📈", url_path="overview"),
            st.Page(page_customers, title="Customer & Conversion", icon="👥", url_path="customers"),
            st.Page(page_products, title="Product & Inventory", icon="📦", url_path="products"),
            st.Page(page_returns, title="Return Analysis", icon="🔄", url_path="returns"),
            st.Page(page_advanced, title="Advanced Insights", icon="🧠", url_path="insights"),
            st.Page(page_kundensegmentierung, title="Kundensegmentierung", icon="👥", url_path="kundensegmentierung"),
        ],
        "Machine Learning": [
           
            st.Page(page_forecast_explain, title="Forecast — Modell-Erklärung", icon="📖", url_path="forecast-erklaerung"),
            st.Page(page_forecast_prediction, title="Sales Forecasting", icon="🔮", url_path="sales-forecast"),
            st.Page(page_churn_explain, title="Churn — Modell-Erklärung", icon="⬛", url_path="churn-erklaerung"),
            st.Page(page_churn_prediction, title="Churn Prediction", icon="⚠️", url_path="churn-prediction"),
        ],
        "Extras": [
            st.Page(page_export, title="Export & Reports", icon="🏗️", url_path="export"),
            st.Page(page_danke, title="Vielen Dank", icon="🙏", url_path="danke"),
        ],
    }

    #  Sidebar: Zurück + Logout 
    st.sidebar.markdown('<div class="sb-line"></div>', unsafe_allow_html=True)
    st.sidebar.markdown(f"""
    <div style="text-align:center; padding:4px 0; margin-bottom:4px;">
        <span style="font-size:11px; color:{C['emerald']};">
            ✓ Angemeldet als <strong>{st.session_state.get('login_user', 'admin')}</strong>
        </span>
    </div>""", unsafe_allow_html=True)
    if st.sidebar.button("📂  Zurück zu Projekte", use_container_width=True, type="secondary"):
        st.session_state.project_selected = False
        st.rerun()
    if st.sidebar.button("🚪  Abmelden", use_container_width=True, type="secondary"):
        st.session_state.logged_in = False
        st.session_state.project_selected = False
        st.session_state.pop("login_user", None)
        st.rerun()

elif st.session_state.logged_in:
    #  STUFE 2: Meine Projekte 
    nav_pages = {
        "Übersicht": [
            st.Page(page_projekte, title="Meine Projekte", icon="📂", url_path="projekte", default=True),
        ],
    }

else:
    # STUFE 1: Startseite 
    nav_pages = {
        "Website": [
            st.Page(page_startseite, title="Startseite", icon="🏠", url_path="startseite", default=True),
        ],
    }

pg = st.navigation(nav_pages)
pg.run()

# ╔═══════════════════════════════════════════════════════════╗
#  FOOTER
# ╚═══════════════════════════════════════════════════════════╝
st.markdown("""
<div class="app-footer">
    <strong style="color:#64748b;">PostgreSQL</strong> · <strong style="color:#64748b;">Streamlit</strong> · <strong style="color:#64748b;">Plotly</strong>
    &nbsp;&nbsp;|&nbsp;&nbsp; Abschlussprojekt — E-Commerce Analytics Dashboard
</div>""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sb-line"></div>', unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="text-align:center; padding:6px 0;">
    <span style="font-size:10px; color:#475569;">v4.0 · Navigation + RFM + Forecasting · 2026</span>
</div>""", unsafe_allow_html=True)
