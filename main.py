"""Entry point for the phishing detection application.

Run it with::

    streamlit run main.py

The module wires together the navigation sidebar and the individual views; all
domain logic lives under ``src/``.
"""

import logging

import streamlit as st

from src.config import STREAMLIT_CONFIG

# Must be the first Streamlit call in the process.
st.set_page_config(
    page_title=STREAMLIT_CONFIG["page_title"],
    page_icon=STREAMLIT_CONFIG["page_icon"],
    layout=STREAMLIT_CONFIG["layout"],
    initial_sidebar_state=STREAMLIT_CONFIG["initial_sidebar_state"],
)

from views.data_exploration import show_data_exploration  # noqa: E402
from views.home import show_home  # noqa: E402
from views.model_performance import show_model_performance  # noqa: E402
from views.prediction import show_prediction  # noqa: E402
from views.preprocessing import show_preprocessing  # noqa: E402
from views.styles import inject_custom_css  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

PAGES = {
    "Home": show_home,
    "Data Exploration": show_data_exploration,
    "Preprocessing": show_preprocessing,
    "Model Performance": show_model_performance,
    "Prediction": show_prediction,
}


def main() -> None:
    """Render the sidebar and dispatch to the selected view."""
    inject_custom_css()

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Choose a section:", list(PAGES))

    PAGES[selection]()

    st.sidebar.markdown("---")
    st.sidebar.caption("Phishing Detection · LightGBM")
    st.sidebar.caption("Version 1.0.0")


if __name__ == "__main__":
    main()
