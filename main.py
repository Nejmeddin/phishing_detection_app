"""
Main phishing detection application.
This application allows analyzing URLs to detect if they are legitimate or malicious.
"""

import streamlit as st
import os
import sys

# Page configuration - DOIT ÊTRE LE PREMIER APPEL À STREAMLIT
st.set_page_config(
    page_title="Phishing Detection",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import des modules après st.set_page_config
from PIL import Image

# Add parent directory to Python path to be able to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import application pages
from pages.home import show_home
from pages.data_exploration import show_data_exploration
from pages.preprocessing import show_preprocessing
from pages.model_performance import show_model_performance
from pages.prediction import show_prediction

# Custom CSS
st.markdown(
    """
<style>
    .main-title {
        font-size: 2.5rem;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.8rem;
        color: #3D85C6;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .info-box {
        background-color: #3D85C6;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #38bdf8;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #3D85C6;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a section:",
    [
        "Home",
        "Data Exploration",
        "Preprocessing",
        "Model Performance",
        "Prediction",
    ],
)

# Display selected page
if page == "Home":
    show_home()
elif page == "Data Exploration":
    show_data_exploration()
elif page == "Preprocessing":
    show_preprocessing()
elif page == "Model Performance":
    show_model_performance()
elif page == "Prediction":
    show_prediction()

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Developed with B-H-N")
st.sidebar.text("Version 1.0.0")
