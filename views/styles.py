"""Shared presentation layer for the Streamlit interface.

The stylesheet lives here rather than in the entry point so that every view
draws from one palette and no page redefines its own copy.
"""

import streamlit as st

CUSTOM_CSS = """
<style>
    :root {
        --pd-primary: #38bdf8;
        --pd-surface: #1e293b;
        --pd-surface-border: #334155;
        --pd-text: #e2e8f0;
        --pd-warning-bg: #422006;
        --pd-warning-border: #f59e0b;
        --pd-success-bg: #052e16;
        --pd-success-border: #22c55e;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--pd-primary);
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.6rem;
        font-weight: 600;
        color: var(--pd-primary);
        margin-top: 2rem;
        margin-bottom: 0.8rem;
    }

    .info-box,
    .warning-box,
    .success-box {
        padding: 1rem 1.25rem;
        border-radius: 8px;
        border-left: 4px solid;
        margin-bottom: 1.25rem;
        color: var(--pd-text);
    }

    .info-box {
        background-color: var(--pd-surface);
        border-left-color: var(--pd-primary);
    }

    .warning-box {
        background-color: var(--pd-warning-bg);
        border-left-color: var(--pd-warning-border);
    }

    .success-box {
        background-color: var(--pd-success-bg);
        border-left-color: var(--pd-success-border);
    }

    .info-box h2,
    .info-box h3,
    .warning-box h3,
    .success-box h3 {
        color: var(--pd-primary);
        margin-top: 0;
        margin-bottom: 0.5rem;
    }

    .warning-box h3 {
        color: var(--pd-warning-border);
    }

    .info-box p,
    .warning-box p,
    .success-box p {
        margin-bottom: 0;
        line-height: 1.6;
    }
</style>
"""


def inject_custom_css() -> None:
    """Apply the application stylesheet to the current Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
