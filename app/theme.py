"""App theming: colors, fonts, and Plotly/Matplotlib defaults."""

import streamlit as st
import plotly.io as pio
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Brand Colors ---
COLORS = {
    "akzent1": "#B02F2C",  # deep red
    "akzent2": "#C43A3B",  # warm red
    "akzent3": "#8AC2D1",  # light blue
    "akzent4": "#262A31",  # dark charcoal
    "akzent5": "#EA9E9E",  # soft pink
    "akzent6": "#C9C9C9",  # light grey
}

# Ordered palette for sequential chart series
COLOR_PALETTE = [
    "#B02F2C", "#8AC2D1", "#C43A3B", "#262A31", "#EA9E9E", "#C9C9C9",
    "#006699", "#FF9900", "#006633", "#CC3333",
]


def inject_custom_css():
    """Inject custom CSS for Futura font and brand styling."""
    st.markdown("""
    <style>
    @import url('https://fonts.cdnfonts.com/css/futura-pt');

    /* Apply Futura globally — exclude icon/svg elements */
    html, body, .stMarkdown, .stMetric, .stButton>button,
    .stSelectbox, .stSlider, .stTextInput, .stNumberInput,
    p, span, label, div, li, td, th {
        font-family: 'Futura PT', 'Futura', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Futura PT', 'Futura', sans-serif !important;
        color: #262A31 !important;
    }

    /* Tab labels */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Futura PT', 'Futura', sans-serif !important;
        font-weight: 500;
    }

    /* Active tab accent */
    .stTabs [aria-selected="true"] {
        border-bottom-color: #B02F2C !important;
    }

    /* Primary button */
    .stButton>button[kind="primary"] {
        background-color: #B02F2C !important;
        border-color: #B02F2C !important;
    }

    /* Metric values */
    [data-testid="stMetricValue"] {
        color: #B02F2C !important;
    }
    </style>
    """, unsafe_allow_html=True)


def setup_plotly_theme():
    """Register and activate a custom Plotly template with brand colors."""
    pio.templates["brand"] = pio.templates["plotly_white"]
    brand = pio.templates["brand"]
    brand.layout.colorway = COLOR_PALETTE
    brand.layout.font = dict(family="Futura PT, Futura, sans-serif", color="#262A31")
    brand.layout.title = dict(font=dict(size=16, color="#262A31"))
    pio.templates.default = "brand"


def setup_matplotlib_theme():
    """Set Matplotlib rcParams for consistent styling."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Futura PT", "Futura", "Helvetica Neue", "Arial"],
        "axes.prop_cycle": plt.cycler(color=COLOR_PALETTE),
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.edgecolor": "#262A31",
        "axes.labelcolor": "#262A31",
        "text.color": "#262A31",
        "xtick.color": "#262A31",
        "ytick.color": "#262A31",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def apply_theme():
    """Apply all theming (call once at app startup)."""
    inject_custom_css()
    setup_plotly_theme()
    setup_matplotlib_theme()
