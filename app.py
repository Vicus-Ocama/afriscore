"""
╔══════════════════════════════════════════════════════════════════╗
║   AFRISCORE  ·  Alternative Credit Intelligence Platform        ║
║   ML-powered creditworthiness for unbanked Africans             ║
║   Author: Okurwoth Vicus Ocama                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="AfriScore · Credit Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ──────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp { background: #0b0f1a; color: #e8eaf0; }

    [data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    [data-testid="stSidebar"] .stSlider > label,
    [data-testid="stSidebar"] .stNumberInput > label,
    [data-testid="stSidebar"] .stSelectbox > label {
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8 !important;
    }

    .sidebar-section {
        background: linear-gradient(135deg, #1e3a5f22, #0f4c7522);
        border: 1px solid #1e40af44;
        border-radius: 8px;
        padding: 10px 14px 4px;
        margin: 12px 0 6px;
    }
    .sidebar-section h4 {
        color: #38bdf8 !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        margin: 0 !important;
    }

    .hero-block {
        background: linear-gradient(135deg, #0f172a 0%, #0d2348 50%, #0a1628 100%);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 36px 40px 28px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .hero-block::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, #3b82f620 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-block::after {
        content: '';
        position: absolute;
        bottom: -40px; left: 20px;
        width: 160px; height: 160px;
        background: radial-gradient(circle, #f59e0b18 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8, #818cf8, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
        line-height: 1.1;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: #94a3b8;
        font-weight: 300;
        max-width: 560px;
        line-height: 1.6;
    }
    .hero-badge {
        display: inline-block;
        background: #1e40af33;
        border: 1px solid #3b82f644;
        color: #93c5fd;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 14px;
    }
    .hero-author {
        margin-top: 14px;
        font-size: 0.8rem;
        color: #475569;
        font-style: italic;
    }
    .hero-author span {
        color: #38bdf8;
        font-style: normal;
        font-weight: 600;
    }

    .metric-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px 22px;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #334155; }
    .metric-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1;
    }
    .metric-sub { font-size: 0.75rem; color: #64748b; margin-top: 4px; }

    .gauge-wrapper {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 28px;
    }

    .decision-approve {
        background: linear-gradient(135deg, #052e16, #14532d);
        border: 1px solid #16a34a66;
        border-left: 4px solid #22c55e;
        border-radius: 12px;
        padding: 18px 24px;
        color: #86efac;
    }
    .decision-review {
        background: linear-gradient(135deg, #1c1003, #451a03);
        border: 1px solid #d97706aa;
        border-left: 4px solid #f59e0b;
        border-radius: 12px;
        padding: 18px 24px;
        color: #fcd34d;
    }
    .decision-reject {
        background: linear-gradient(135deg, #1c0303, #450a0a);
        border: 1px solid #dc262666;
        border-left: 4px solid #ef4444;
        border-radius: 12px;
        padding: 18px 24px;
        color: #fca5a5;
    }
    .decision-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .decision-text { font-size: 0.88rem; line-height: 1.5; opacity: 0.9; }

    .section-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #e2e8f0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 28px 0 14px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-title::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, #334155, transparent);
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin: 14px 0;
    }
    .feature-item {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 12px 14px;
    }
    .feature-name {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 4px;
    }
    .feature-val {
        font-family: 'Syne', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: #e2e8f0;
    }

    .risk-pill-red {
        display: inline-block;
        background: #450a0a88;
        border: 1px solid #ef444466;
        color: #fca5a5;
        font-size: 0.75rem;
        padding: 4px 10px;
        border-radius: 20px;
        margin: 3px;
    }
    .risk-pill-green {
        display: inline-block;
        background: #052e1688;
        border: 1px solid #22c55e66;
        color: #86efac;
        font-size: 0.75rem;
        padding: 4px 10px;
        border-radius: 20px;
        margin: 3px;
    }
    .risk-pill-yellow {
        display: inline-block;
        background: #451a0388;
        border: 1px solid #f59e0b66;
        color: #fcd34d;
        font-size: 0.75rem;
        padding: 4px 10px;
        border-radius: 20px;
        margin: 3px;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: #111827;
        border-radius: 10px;
        padding: 4px;
        gap: 2px;
        border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        color: #64748b !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.83rem !important;
        font-weight: 500 !important;
        padding: 8px 18px !important;
        border-radius: 7px !important;
    }
    .stTabs [aria-selected="true"] {
        background: #1e3a5f !important;
        color: #38bdf8 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.05em !important;
        padding: 14px 0 !important;
        transition: all 0.2s !important;
        box-shadow: 0 4px 20px #1d4ed840 !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
        box-shadow: 0 6px 28px #3b82f640 !important;
        transform: translateY(-1px) !important;
    }

    hr { border-color: #1e293b !important; }

    .streamlit-expanderHeader {
        background: #111827 !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
    }

    .footer-block {
        text-align: center;
        padding: 24px 0 12px;
        color: #334155;
        font-size: 0.75rem;
        border-top: 1px solid #1e293b;
        margin-top: 40px;
    }

    .info-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 0.83rem;
        color: #94a3b8;
        line-height: 1.6;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Load model ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("models/lgbm_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/feature_cols.pkl", "rb") as f:
            cols = pickle.load(f)
        return model, cols
    except FileNotFoundError:
        return None, None


model, feature_cols = load_model()


# ── Helper: score → gauge angle ────────────────────────────────
def s2t(s: float) -> float:
    """Map credit score (300–850) to radians on a half-circle gauge."""
    return np.pi - (s - 300) / (850 - 300) * np.pi


# ── Helper: Gauge chart ─────────────────────────────────────────
def draw_gauge(score: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    bands = [
        (300, 549, "#ef444444", "#ef4444"),
        (550, 699, "#f59e0b44", "#f59e0b"),
        (700, 850, "#22c55e44", "#22c55e"),
    ]

    for lo, hi, bg, stroke in bands:
        t1, t2 = s2t(lo), s2t(hi)
        theta = np.linspace(t2, t1, 120)
        outer, inner = 1.0, 0.68
        xs = np.concatenate([outer * np.cos(theta), inner * np.cos(theta[::-1])])
        ys = np.concatenate([outer * np.sin(theta), inner * np.sin(theta[::-1])])
        ax.fill(xs, ys, color=bg, zorder=1)
        ax.plot(outer * np.cos(theta), outer * np.sin(theta), color=stroke, lw=2, zorder=2)
        ax.plot(inner * np.cos(theta), inner * np.sin(theta), color=stroke, lw=0.5, alpha=0.4, zorder=2)

    # Needle
    needle_theta = s2t(score)
    nx, ny = 0.82 * np.cos(needle_theta), 0.82 * np.sin(needle_theta)
    ax.annotate(
        "", xy=(nx, ny), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="#f8fafc", lw=2.5, mutation_scale=16),
    )
    ax.add_patch(plt.Circle((0, 0), 0.07, color="#38bdf8", zorder=5))

    # Band labels
    for s, label in [(300, "300"), (575, "Moderate"), (700, "700"), (850, "850")]:
        t = s2t(s)
        ax.text(
            1.15 * np.cos(t), 1.15 * np.sin(t), label,
            ha="center", va="center", fontsize=6.5, color="#64748b",
        )

    # Score colour
    sc = "#22c55e" if score >= 700 else "#f59e0b" if score >= 550 else "#ef4444"
    ax.text(0, -0.22, str(score), ha="center", va="center",
            fontsize=28, fontweight="bold", color=sc)
    ax.text(0, -0.42, "CREDIT SCORE", ha="center", va="center",
            fontsize=7, color="#64748b", fontweight="bold")

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.6, 1.2)
    ax.axis("off")
    plt.tight_layout(pad=0.1)
    return fig


# ── Helper: Payment timeline chart ─────────────────────────────
def draw_payment_timeline(pay_vals, bill_vals, pay_amt_vals):
    months = ["Month 6", "Month 5", "Month 4", "Month 3", "Month 2", "Last"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    fig.patch.set_facecolor("#111827")

    # Delay status bar chart
    ax = axes[0]
    ax.set_facecolor("#111827")
    colors = ["#ef4444" if p > 0 else "#22c55e" for p in pay_vals]
    ax.bar(months, pay_vals, color=colors, width=0.55, zorder=3)
    ax.axhline(0, color="#334155", lw=1, zorder=2)
    ax.tick_params(colors="#64748b", labelsize=7)
    ax.spines[:].set_visible(False)
    ax.set_title("Payment Delay Status", color="#94a3b8", fontsize=8, pad=8)
    ax.set_ylabel("Delay (months)", color="#64748b", fontsize=7)
    ax.grid(axis="y", color="#1e293b", lw=0.8, zorder=0)

    # Bill vs paid grouped bar chart
    ax2 = axes[1]
    ax2.set_facecolor("#111827")
    x = np.arange(len(months))
    w = 0.35
    ax2.bar(x - w / 2, bill_vals, w, color="#3b82f644", edgecolor="#3b82f6", label="Bill", zorder=3)
    ax2.bar(x + w / 2, pay_amt_vals, w, color="#22c55e44", edgecolor="#22c55e", label="Paid", zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(months, fontsize=7, color="#64748b")
    ax2.tick_params(colors="#64748b", labelsize=7)
    ax2.spines[:].set_visible(False)
    ax2.set_title("Bill Amount vs Payment Made", color="#94a3b8", fontsize=8, pad=8)
    ax2.set_ylabel("Amount (UGX)", color="#64748b", fontsize=7)
    ax2.grid(axis="y", color="#1e293b", lw=0.8, zorder=0)
    ax2.legend(fontsize=7, framealpha=0, labelcolor="#94a3b8")

    plt.tight_layout()
    return fig


# ── SIDEBAR ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 10px 0 18px;'>
            <span style='font-family:Syne,sans-serif; font-size:1.35rem;
                         font-weight:800; color:#38bdf8;'>🌍 AfriScore</span><br>
            <span style='font-size:0.7rem; color:#475569; letter-spacing:0.08em;
                         text-transform:uppercase;'>Credit Intelligence Platform</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section"><h4>👤 Applicant Profile</h4></div>',
                unsafe_allow_html=True)
    LIMIT_BAL = st.number_input("Credit Limit (UGX)", 1_000_000, 500_000_000, 50_000_000, step=1_000_000)
    col_s, col_e = st.columns(2)
    with col_s:
        SEX = st.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    with col_e:
        MARRIAGE = st.selectbox(
            "Marital Status", [1, 2, 3],
            format_func=lambda x: {1: "Married", 2: "Single", 3: "Other"}[x],
        )
    EDUCATION = st.selectbox(
        "Education Level", [1, 2, 3, 4],
        format_func=lambda x: {1: "Graduate", 2: "University", 3: "High School", 4: "Other"}[x],
    )
    AGE = st.slider("Age", 21, 79, 35)

    st.markdown('<div class="sidebar-section"><h4>📅 Payment History (6 months)</h4></div>',
                unsafe_allow_html=True)
    st.caption("–2 No consumption · –1 Paid in full · 0 Revolving · 1–8 Months late")
    PAY_0 = st.slider("Last month",    -2, 8, 0, key="p0")
    PAY_2 = st.slider("2 months ago",  -2, 8, 0, key="p2")
    PAY_3 = st.slider("3 months ago",  -2, 8, 0, key="p3")
    PAY_4 = st.slider("4 months ago",  -2, 8, 0, key="p4")
    PAY_5 = st.slider("5 months ago",  -2, 8, 0, key="p5")
    PAY_6 = st.slider("6 months ago",  -2, 8, 0, key="p6")

    st.markdown('<div class="sidebar-section"><h4>🧾 Bill Amounts (UGX)</h4></div>',
                unsafe_allow_html=True)
    BILL_AMT1 = st.number_input("Last month bill",  0, 500_000_000, 10_000_000, key="b1")
    BILL_AMT2 = st.number_input("2 months ago",     0, 500_000_000, 10_000_000, key="b2")
    BILL_AMT3 = st.number_input("3 months ago",     0, 500_000_000,  8_000_000, key="b3")
    BILL_AMT4 = st.number_input("4 months ago",     0, 500_000_000,  8_000_000, key="b4")
    BILL_AMT5 = st.number_input("5 months ago",     0, 500_000_000,  6_000_000, key="b5")
    BILL_AMT6 = st.number_input("6 months ago",     0, 500_000_000,  6_000_000, key="b6")

    st.markdown('<div class="sidebar-section"><h4>💰 Payments Made (UGX)</h4></div>',
                unsafe_allow_html=True)
    PAY_AMT1 = st.number_input("Last month paid",   0, 500_000_000, 5_000_000, key="pa1")
    PAY_AMT2 = st.number_input("2 months ago paid", 0, 500_000_000, 5_000_000, key="pa2")
    PAY_AMT3 = st.number_input("3 months ago paid", 0, 500_000_000, 3_000_000, key="pa3")
    PAY_AMT4 = st.number_input("4 months ago paid", 0, 500_000_000, 3_000_000, key="pa4")
    PAY_AMT5 = st.number_input("5 months ago paid", 0, 500_000_000, 2_000_000, key="pa5")
    PAY_AMT6 = st.number_input("6 months ago paid", 0, 500_000_000, 2_000_000, key="pa6")

    st.markdown("<br>", unsafe_allow_html=True)
    assess_btn = st.button("🔍  Run Credit Assessment", use_container_width=True)


# ── Feature engineering ─────────────────────────────────────────
pay_vals     = [PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6]
bill_vals    = [BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6]
pay_amt_vals = [PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6]

delay_count = sum(1 for p in pay_vals if p > 0)
max_delay   = max(pay_vals)
avg_bill    = float(np.mean(bill_vals))
avg_pay_amt = float(np.mean(pay_amt_vals))
pay_ratio   = avg_pay_amt / (avg_bill + 1)
utilization = BILL_AMT1 / (LIMIT_BAL + 1)
bill_trend  = BILL_AMT1 - BILL_AMT6

input_dict = {
    "LIMIT_BAL": LIMIT_BAL, "SEX": SEX, "EDUCATION": EDUCATION,
    "MARRIAGE": MARRIAGE, "AGE": AGE,
    "PAY_0": PAY_0, "PAY_2": PAY_2, "PAY_3": PAY_3,
    "PAY_4": PAY_4, "PAY_5": PAY_5, "PAY_6": PAY_6,
    "BILL_AMT1": BILL_AMT1, "BILL_AMT2": BILL_AMT2, "BILL_AMT3": BILL_AMT3,
    "BILL_AMT4": BILL_AMT4, "BILL_AMT5": BILL_AMT5, "BILL_AMT6": BILL_AMT6,
    "PAY_AMT1": PAY_AMT1, "PAY_AMT2": PAY_AMT2, "PAY_AMT3": PAY_AMT3,
    "PAY_AMT4": PAY_AMT4, "PAY_AMT5": PAY_AMT5, "PAY_AMT6": PAY_AMT6,
    "delay_count": delay_count, "max_delay": max_delay,
    "avg_bill": avg_bill, "avg_pay_amt": avg_pay_amt,
    "pay_ratio": pay_ratio, "utilization": utilization, "bill_trend": bill_trend,
}

# Only build input_df when model is available (avoids crash when feature_cols is None)
input_df = pd.DataFrame([input_dict])[feature_cols] if feature_cols is not None else None


# ── MAIN CONTENT ─────────────────────────────────────────────────

# Hero block — includes author byline
st.markdown(
    """
    <div class="hero-block">
        <div class="hero-badge">🌍 Sub-Saharan Africa · Financial Inclusion</div>
        <div class="hero-title">AfriScore Intelligence</div>
        <div class="hero-sub">
            Machine learning–powered credit assessment for the 57% of Sub-Saharan Africans
            excluded from traditional banking. Using LightGBM + SHAP explainability on
            30,000+ applicant records — fair, transparent, and fast.
        </div>
        <div class="hero-author">Built by <span>Okurwoth Vicus Ocama</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Stats row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">Training Records</div>'
        '<div class="metric-value" style="color:#38bdf8;">30K</div>'
        '<div class="metric-sub">Applicant profiles</div></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">Model ROC-AUC</div>'
        '<div class="metric-value" style="color:#818cf8;">0.78</div>'
        '<div class="metric-sub">LightGBM classifier</div></div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">Features</div>'
        '<div class="metric-value" style="color:#f472b6;">30</div>'
        '<div class="metric-sub">Engineered signals</div></div>',
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">Explainability</div>'
        '<div class="metric-value" style="color:#34d399;">SHAP</div>'
        '<div class="metric-sub">Decision transparency</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ── TABS ─────────────────────────────────────────────────────────
tab_overview, tab_result, tab_explain, tab_about = st.tabs(
    ["📊  Live Preview", "🎯  Assessment Result", "🔬  SHAP Explainability", "ℹ️  About"]
)


# ─── TAB 1 : Live Preview ────────────────────────────────────────
with tab_overview:
    st.markdown('<div class="section-title">📋 Applicant Summary</div>', unsafe_allow_html=True)

    ed_map  = {1: "Graduate", 2: "University", 3: "High School", 4: "Other"}
    mar_map = {1: "Married", 2: "Single", 3: "Other"}

    st.markdown(
        f"""
        <div class="feature-grid">
            <div class="feature-item"><div class="feature-name">Credit Limit</div>
                <div class="feature-val">UGX {LIMIT_BAL:,}</div></div>
            <div class="feature-item"><div class="feature-name">Age</div>
                <div class="feature-val">{AGE} yrs</div></div>
            <div class="feature-item"><div class="feature-name">Education</div>
                <div class="feature-val">{ed_map[EDUCATION]}</div></div>
            <div class="feature-item"><div class="feature-name">Marital Status</div>
                <div class="feature-val">{mar_map[MARRIAGE]}</div></div>
            <div class="feature-item"><div class="feature-name">Delay Count</div>
                <div class="feature-val">{delay_count} / 6 months</div></div>
            <div class="feature-item"><div class="feature-name">Max Delay</div>
                <div class="feature-val">{max_delay} months</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">📈 Engineered Risk Signals</div>', unsafe_allow_html=True)

    util_pct   = utilization * 100
    util_color = "#ef4444" if util_pct > 80 else "#f59e0b" if util_pct > 50 else "#22c55e"
    pay_r_color = "#ef4444" if pay_ratio < 0.2 else "#f59e0b" if pay_ratio < 0.5 else "#22c55e"
    trend_color = "#ef4444" if bill_trend > 2000 else "#f59e0b" if bill_trend > 0 else "#22c55e"

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Credit Utilization</div>'
            f'<div class="metric-value" style="color:{util_color};">{util_pct:.1f}%</div>'
            f'<div class="metric-sub">Bill ÷ Limit ratio</div></div>',
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Pay-to-Bill Ratio</div>'
            f'<div class="metric-value" style="color:{pay_r_color};">{pay_ratio:.2f}</div>'
            f'<div class="metric-sub">Avg payment ÷ Avg bill</div></div>',
            unsafe_allow_html=True,
        )
    with r3:
        trend_label = f"+UGX {bill_trend:,.0f}" if bill_trend >= 0 else f"UGX {bill_trend:,.0f}"
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Bill Trend</div>'
            f'<div class="metric-value" style="color:{trend_color}; font-size:1.5rem;">{trend_label}</div>'
            f'<div class="metric-sub">Last vs 6 months ago</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">📅 6-Month Financial Timeline</div>', unsafe_allow_html=True)
    fig_timeline = draw_payment_timeline(pay_vals, bill_vals, pay_amt_vals)
    st.pyplot(fig_timeline, use_container_width=True)
    plt.close(fig_timeline)

    st.markdown('<div class="section-title">⚠️ Risk Signal Summary</div>', unsafe_allow_html=True)
    pills_html = ""
    if delay_count >= 3:
        pills_html += '<span class="risk-pill-red">⛔ Frequent delays</span>'
    elif delay_count > 0:
        pills_html += '<span class="risk-pill-yellow">⚠️ Some delays detected</span>'
    else:
        pills_html += '<span class="risk-pill-green">✅ No payment delays</span>'

    if util_pct > 80:
        pills_html += '<span class="risk-pill-red">⛔ High utilization</span>'
    elif util_pct > 50:
        pills_html += '<span class="risk-pill-yellow">⚠️ Moderate utilization</span>'
    else:
        pills_html += '<span class="risk-pill-green">✅ Low utilization</span>'

    if pay_ratio >= 0.5:
        pills_html += '<span class="risk-pill-green">✅ Good repayment rate</span>'
    elif pay_ratio >= 0.2:
        pills_html += '<span class="risk-pill-yellow">⚠️ Partial repayments</span>'
    else:
        pills_html += '<span class="risk-pill-red">⛔ Low repayment rate</span>'

    if bill_trend > 2_000_000:
        pills_html += '<span class="risk-pill-red">⛔ Rising debt</span>'
    elif bill_trend < 0:
        pills_html += '<span class="risk-pill-green">✅ Decreasing debt</span>'
    else:
        pills_html += '<span class="risk-pill-yellow">⚠️ Stable debt</span>'

    st.markdown(f"<div>{pills_html}</div>", unsafe_allow_html=True)
    st.markdown(
        "<br><div class='info-box'>💡 These signals update in real-time as you adjust the sliders. "
        "Click <strong>Run Credit Assessment</strong> in the sidebar to generate the ML prediction "
        "and SHAP explanation.</div>",
        unsafe_allow_html=True,
    )


# ─── TAB 2 : Assessment Result ───────────────────────────────────
with tab_result:
    if not assess_btn:
        st.markdown(
            """
            <div style='text-align:center; padding:60px 20px;'>
                <div style='font-size:3.5rem; margin-bottom:16px;'>🎯</div>
                <div style='font-family:Syne,sans-serif; font-size:1.2rem;
                             font-weight:700; color:#e2e8f0; margin-bottom:8px;'>
                    Ready to Assess
                </div>
                <div style='color:#64748b; font-size:0.9rem; max-width:360px; margin:0 auto;'>
                    Configure the applicant profile in the sidebar, then click
                    <strong style='color:#38bdf8'>Run Credit Assessment</strong> to generate
                    the score, decision, and full risk breakdown.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        if model is None or input_df is None:
            st.error(
                "⚠️ Model files not found. Make sure `models/lgbm_model.pkl` and "
                "`models/feature_cols.pkl` are present in the project directory."
            )
        else:
            prob_default = model.predict_proba(input_df)[0][1]
            credit_score = int((1 - prob_default) * 850)
            credit_score = max(300, min(850, credit_score))

            st.markdown('<div class="section-title">🎯 Credit Score</div>', unsafe_allow_html=True)
            gcol, dcol = st.columns([1, 1.4])

            with gcol:
                st.markdown('<div class="gauge-wrapper">', unsafe_allow_html=True)
                fig_gauge = draw_gauge(credit_score)
                st.pyplot(fig_gauge, use_container_width=True)
                plt.close(fig_gauge)
                st.markdown("</div>", unsafe_allow_html=True)

            with dcol:
                prob_pct = f"{prob_default:.1%}"
                safe_pct = f"{(1 - prob_default):.1%}"
                m1, m2 = st.columns(2)
                with m1:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-label">Default Risk</div>'
                        f'<div class="metric-value" style="color:#ef4444;">{prob_pct}</div>'
                        f'<div class="metric-sub">Probability of default</div></div>',
                        unsafe_allow_html=True,
                    )
                with m2:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-label">Creditworthiness</div>'
                        f'<div class="metric-value" style="color:#22c55e;">{safe_pct}</div>'
                        f'<div class="metric-sub">Repayment likelihood</div></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                if credit_score >= 700:
                    tier, label, detail = "approve", "✅ LOW RISK — Approved", (
                        "Strong payment history, controlled utilization, and consistent repayments. "
                        f"Score of <b>{credit_score}</b> places this applicant in the prime segment. "
                        "Recommend full approval."
                    )
                elif credit_score >= 550:
                    tier, label, detail = "review", "⚠️ MEDIUM RISK — Manual Review", (
                        f"Score of <b>{credit_score}</b> indicates moderate risk. Some payment delays or "
                        "elevated utilization detected. A loan officer review is recommended before approval."
                    )
                else:
                    tier, label, detail = "reject", "❌ HIGH RISK — Not Recommended", (
                        f"Score of <b>{credit_score}</b> reflects significant payment issues and/or high "
                        "debt utilization. Default probability exceeds acceptable threshold. "
                        "Consider alternative products or a smaller credit facility."
                    )

                st.markdown(
                    f'<div class="decision-{tier}"><div class="decision-title">{label}</div>'
                    f'<div class="decision-text">{detail}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="section-title">📊 Score Band Reference</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div style='display:flex; gap:10px; margin:6px 0;'>
                    <div style='flex:1; background:#450a0a55; border:1px solid #ef444466;
                         border-radius:8px; padding:14px; text-align:center;'>
                        <div style='color:#ef4444; font-family:Syne,sans-serif; font-weight:700;
                             font-size:1.1rem;'>300 – 549</div>
                        <div style='color:#fca5a5; font-size:0.8rem; margin-top:4px;'>High Risk</div>
                        <div style='color:#64748b; font-size:0.73rem;'>Not recommended</div>
                    </div>
                    <div style='flex:1; background:#451a0355; border:1px solid #f59e0b66;
                         border-radius:8px; padding:14px; text-align:center;'>
                        <div style='color:#f59e0b; font-family:Syne,sans-serif; font-weight:700;
                             font-size:1.1rem;'>550 – 699</div>
                        <div style='color:#fcd34d; font-size:0.8rem; margin-top:4px;'>Medium Risk</div>
                        <div style='color:#64748b; font-size:0.73rem;'>Manual review</div>
                    </div>
                    <div style='flex:1; background:#05150e55; border:1px solid #22c55e66;
                         border-radius:8px; padding:14px; text-align:center;'>
                        <div style='color:#22c55e; font-family:Syne,sans-serif; font-weight:700;
                             font-size:1.1rem;'>700 – 850</div>
                        <div style='color:#86efac; font-size:0.8rem; margin-top:4px;'>Low Risk</div>
                        <div style='color:#64748b; font-size:0.73rem;'>Recommend approval</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Persist result for the SHAP tab
            st.session_state["result"] = {
                "score": credit_score,
                "prob": prob_default,
                "input_df": input_df,
            }


# ─── TAB 3 : SHAP Explainability ────────────────────────────────
with tab_explain:
    result = st.session_state.get("result")

    if result is None:
        st.markdown(
            """
            <div style='text-align:center; padding:60px 20px;'>
                <div style='font-size:3.5rem; margin-bottom:16px;'>🔬</div>
                <div style='font-family:Syne,sans-serif; font-size:1.2rem;
                             font-weight:700; color:#e2e8f0; margin-bottom:8px;'>
                    Run Assessment First
                </div>
                <div style='color:#64748b; font-size:0.9rem;'>
                    SHAP explanation will appear here after running the credit assessment.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif model is None:
        st.error("Model not loaded.")
    else:
        _input_df = result["input_df"]

        st.markdown('<div class="section-title">🔬 SHAP Feature Contribution</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">SHAP (SHapley Additive exPlanations) shows <b>why</b> '
            'the model reached this score. Red bars push the risk <em>higher</em>; '
            'blue bars push it <em>lower</em>. Every decision is fully auditable and fair.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        with st.spinner("Computing SHAP values…"):
            explainer  = shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(_input_df)

            # LightGBM binary: shap_values may be a list [neg_class, pos_class] or a 2-D array.
            # We always want the positive-class (default=1) values.
            if isinstance(shap_vals, list):
                sv = shap_vals[1][0]          # list of two arrays → take class-1 row 0
            else:
                sv = shap_vals[0]             # single 2-D array → take row 0

            # Expected value: same dual-format handling
            ev = (
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            )

            fig_shap, _ = plt.subplots(figsize=(9, 5))
            fig_shap.patch.set_facecolor("#111827")
            plt.gca().set_facecolor("#111827")

            shap.waterfall_plot(
                shap.Explanation(
                    values=sv,
                    base_values=ev,
                    data=_input_df.iloc[0],
                    feature_names=feature_cols,
                ),
                show=False,
                max_display=15,
            )
            plt.gcf().set_facecolor("#111827")
            plt.gca().set_facecolor("#111827")
            for item in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
                item.set_color("#94a3b8")
            plt.gca().spines[:].set_color("#334155")
            plt.title(
                "SHAP Waterfall — Feature Impact on This Prediction",
                color="#e2e8f0", fontsize=9, pad=10,
            )
            plt.tight_layout()
            st.pyplot(fig_shap, use_container_width=True)
            plt.close(fig_shap)

        # Top features table
        st.markdown('<div class="section-title">📋 Top Feature Contributions</div>', unsafe_allow_html=True)
        shap_series = pd.Series(sv, index=feature_cols)
        top_n = shap_series.abs().nlargest(10)
        top_df = pd.DataFrame({
            "Feature":     top_n.index,
            "SHAP Value":  shap_series[top_n.index].round(4),
            "Direction":   shap_series[top_n.index].apply(
                               lambda v: "⬆️ Increases Risk" if v > 0 else "⬇️ Reduces Risk"
                           ),
            "Input Value": _input_df.iloc[0][top_n.index].round(2).values,
        }).reset_index(drop=True)

        st.dataframe(
            top_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "SHAP Value":  st.column_config.NumberColumn(format="%.4f"),
                "Input Value": st.column_config.NumberColumn(format="%.2f"),
            },
        )

        with st.expander("ℹ️ How to read SHAP values"):
            st.markdown(
                """
                - **Positive SHAP** → this feature *increases* predicted default probability (worsens score)
                - **Negative SHAP** → this feature *decreases* predicted default probability (improves score)
                - The **base value** is the model's average prediction across all training data
                - Final prediction = base value + sum of all SHAP contributions
                - This approach satisfies **GDPR Article 22** requirements for algorithmic explainability
                """
            )


# ─── TAB 4 : About ──────────────────────────────────────────────
with tab_about:
    st.markdown(
        """
        <div style='max-width:680px;'>
        <div class="section-title">🌍 Project Context</div>
        <div class="info-box">
            An estimated <strong>57% of Sub-Saharan Africans</strong> lack access to formal banking.
            Without a credit history, millions are excluded from loans, housing, and capital that
            could transform their lives. Traditional FICO-style scoring requires bank statements
            and formal employment records — data that the unbanked simply don't have.<br><br>
            <strong>AfriScore</strong> demonstrates how alternative data signals — payment behaviour,
            utilization trends, and repayment consistency — can proxy creditworthiness, opening
            access to finance for the previously excluded.
        </div>

        <div class="section-title">🤖 Model Architecture</div>
        <div class="info-box">
            <strong>Algorithm:</strong> LightGBM (Gradient Boosted Trees)<br>
            <strong>Training data:</strong> 30,000 credit card applicants (UCI Credit Default Dataset)<br>
            <strong>Features:</strong> 30 engineered signals including payment delays, utilization,
            pay-to-bill ratio, and bill trend<br>
            <strong>Evaluation:</strong> ROC-AUC, Precision-Recall, Calibration curves<br>
            <strong>Explainability:</strong> SHAP TreeExplainer for per-prediction transparency<br>
            <strong>Score scaling:</strong> Default probability mapped to 300–850 credit score range
        </div>

        <div class="section-title">⚖️ Fairness & Ethics</div>
        <div class="info-box">
            Credit models carry real-world consequences. This project incorporates:<br><br>
            • <strong>SHAP explainability</strong> — every decision is auditable<br>
            • <strong>Demographic parity checks</strong> — gender/education bias monitored<br>
            • <strong>Manual review tier</strong> — borderline cases escalated to humans<br>
            • <strong>Transparent score bands</strong> — no black-box approvals<br><br>
            Responsible AI in lending is not optional — it's a regulatory and ethical requirement.
        </div>

        <div class="section-title">🛠️ Tech Stack</div>
        <div class="feature-grid">
            <div class="feature-item"><div class="feature-name">ML Framework</div>
                <div class="feature-val" style="font-size:0.9rem;">LightGBM</div></div>
            <div class="feature-item"><div class="feature-name">Explainability</div>
                <div class="feature-val" style="font-size:0.9rem;">SHAP</div></div>
            <div class="feature-item"><div class="feature-name">Frontend</div>
                <div class="feature-val" style="font-size:0.9rem;">Streamlit</div></div>
            <div class="feature-item"><div class="feature-name">Data</div>
                <div class="feature-val" style="font-size:0.9rem;">pandas / numpy</div></div>
            <div class="feature-item"><div class="feature-name">Visualisation</div>
                <div class="feature-val" style="font-size:0.9rem;">Matplotlib</div></div>
            <div class="feature-item"><div class="feature-name">Language</div>
                <div class="feature-val" style="font-size:0.9rem;">Python 3.13</div></div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Footer ───────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer-block">
        AfriScore · Alternative Credit Intelligence · Built with LightGBM + SHAP + Streamlit<br>
        <span style='color:#475569;'>
            By <strong style='color:#38bdf8;'>Okurwoth Vicus Ocama</strong> ·
            Data: UCI Credit Default Dataset · 30,000 records
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)