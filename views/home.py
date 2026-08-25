"""
Home page of the phishing detection application.
Provides an overview of the app and its features.
"""

import streamlit as st

# A shield with a fish hook through it: protection against phishing. Inlined as
# SVG so it scales cleanly and needs no asset pipeline or network access.
LOGO_SVG = """
<div style="display:flex;justify-content:center;margin:0.5rem 0 2rem;">
<svg width="120" height="120" viewBox="0 0 120 120" role="img"
     aria-label="Phishing detection shield logo">
  <defs>
    <linearGradient id="shieldFill" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#38bdf8"/>
      <stop offset="100%" stop-color="#0369a1"/>
    </linearGradient>
  </defs>
  <path d="M60 10 L102 26 V60 C102 84 84 102 60 110 C36 102 18 84 18 60 V26 Z"
        fill="url(#shieldFill)" stroke="#7dd3fc" stroke-width="2.5"/>
  <path d="M74 34 V62 A16 16 0 0 1 42 62 A16 16 0 0 1 52 47"
        fill="none" stroke="#0f172a" stroke-width="7"
        stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M74 34 L66 42" stroke="#0f172a" stroke-width="7"
        stroke-linecap="round"/>
  <circle cx="74" cy="32" r="4.5" fill="#0f172a"/>
</svg>
</div>
"""


def show_home():
    """Displays the home page of the application."""

    # Main title
    st.markdown(
        "<h1 class='main-title'>Phishing Detection with LightGBM</h1>",
        unsafe_allow_html=True,
    )

    # Logo, inlined rather than fetched from a CDN so the page renders
    # identically offline and carries no third-party dependency.
    st.markdown(LOGO_SVG, unsafe_allow_html=True)

    # Introduction
    st.markdown(
        """
    <div class='info-box'>
    <h2>Welcome to our phishing detection app!</h2>
    <p>This application uses an advanced machine learning model (LightGBM) to analyze URLs 
    and determine whether they are legitimate or malicious (phishing).</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # About phishing
    st.markdown(
        "<h2 class='section-title'>What is phishing?</h2>",
        unsafe_allow_html=True,
    )
    st.write(
        """
    Phishing is a fraudulent technique aimed at tricking users into giving away sensitive information 
    (credentials, passwords, banking data, etc.) by pretending to be a trusted entity.
    
    Phishing attacks often use:
    - URLs that look like legitimate websites but with subtle differences
    - Social engineering techniques to create a sense of urgency
    - Websites that mimic the appearance of known platforms
    """
    )

    # App features
    st.markdown(
        "<h2 class='section-title'>Application Features</h2>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **Exploration and analysis:**
        - 📊 Data distribution visualization
        - 📉 Descriptive statistics
        - 🔄 Explanation of preprocessing steps
        - 🔍 Analysis of variable correlations
        """
        )

    with col2:
        st.markdown(
            """
        **Model and predictions:**
        - 📈 Model performance visualization
        - 🎯 Detailed accuracy metrics
        - 🔮 Real-time prediction on URLs
        - 🛡️ Personalized security tips
        """
        )

    # How to use the app
    st.markdown(
        "<h2 class='section-title'>How to use this application?</h2>",
        unsafe_allow_html=True,
    )
    st.write(
        """
    1. Use the sidebar to navigate between sections
    2. Explore the data and preprocessing steps to understand key features
    3. Check the model’s performance to evaluate its reliability
    4. Go to the prediction page to analyze a URL in real time
    """
    )

    # Warning
    st.markdown(
        """
    <div class='warning-box'>
    <h3>⚠️ Warning</h3>
    <p>This application is a decision-support tool and does not replace common sense and good online 
    security practices. Even with a high-performance model, some sophisticated attacks may not be detected.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Learn more
    st.markdown("<h2 class='section-title'>Learn More</h2>", unsafe_allow_html=True)
    st.write(
        """
    This application uses a LightGBM model trained on a dataset of legitimate and malicious URLs.
    The model analyzes over 30 features extracted from URLs to make its predictions.
    
    For more information:
    - Check out the "Data Exploration" and "Preprocessing" sections to understand the data used
    - Visit the "Model Performance" section to see how the model was evaluated
    - Test your own URLs in the "Prediction" section
    """
    )
