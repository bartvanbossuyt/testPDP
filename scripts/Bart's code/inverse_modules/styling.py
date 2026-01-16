# -*- coding: utf-8 -*-
"""
Styling Module

Handles page configuration and CSS styling for the PDP inverse application.
Provides academic look with consistent formatting across the app.
"""

import streamlit as st


def setup_page_config():
    """
    Configure Streamlit page settings.
    
    Sets page title, icon, layout, and sidebar state.
    Must be called before any other Streamlit commands.
    """
    st.set_page_config(
        page_title="pdp inverse",
        page_icon="📐",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def apply_custom_css():
    """
    Apply custom CSS styling for academic look.
    
    Includes:
    - Academic serif fonts (Georgia, Times New Roman)
    - Consistent spacing and margins
    - Settings card styling
    - Equal width columns for plots
    - LaTeX formula formatting
    """
    st.markdown(
        """
<style>
.block-container { padding: 1rem 1.2rem; max-width: 1800px; }
html, body, [class*='css'] { font-family: "Georgia","Times New Roman",serif; color:#111; }
.figure-title { font-size:1.00rem; font-weight:600; letter-spacing:.2px; margin-bottom:.4rem; }
h1, .headline { font-weight:700; letter-spacing:.5px; margin-bottom:.6rem; }
hr { border:none; border-top:1px solid #ddd; margin:.4rem 0 1rem 0; }
/* settings card */
.settings-card {
    background: #fafafa;
    border: 1px solid #e6e6e6;
    border-radius: 8px;
    padding: 0.6rem 0.8rem 0.2rem 0.8rem;
    margin: 0.3rem 0 0.8rem 0;
}
.settings-card h3 { font-size: 1.0rem; margin: 0 0 0.3rem 0; font-weight: 600; }
/* Force both plot columns to have identical width */
[data-testid="stHorizontalBlock"] > [data-testid="column"] {
    width: calc(50% - 0.5rem) !important;
    flex: 0 0 calc(50% - 0.5rem) !important;
}
/* Force matplotlib figures to have same size in both columns */
[data-testid="column"] [data-testid="stImage"],
[data-testid="column"] .stPlotlyChart,
[data-testid="column"] > div > div > img {
    max-width: 100% !important;
    width: 100% !important;
}
/* LaTeX formulas should not overflow and have fixed height */
.stLatex {
    overflow-x: auto !important;
    max-width: 100% !important;
    min-height: 2em !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def show_header():
    """Display the main page header with title and divider."""
    st.markdown("<h1 class='headline'>pdp inverse</h1>", unsafe_allow_html=True)
    st.markdown("<hr />", unsafe_allow_html=True)
