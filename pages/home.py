"""
Home page of the phishing detection application.
Provides an overview of the app and its features.
"""

import streamlit as st


def show_home():
    """Displays the home page of the application."""

    # st.markdown(
    #     """
    #     <style>
    #     .info-box {
    #         background-color: #1e293b;
    #         color: #f8fafc;
    #         padding: 20px;
    #         border-radius: 12px;
    #         box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    #         font-family: 'Segoe UI', sans-serif;
    #     }

    #     .info-box h2 {
    #         color: #38bdf8;
    #         margin-bottom: 10px;
    #     }

    #     .info-box p {
    #         color: #e2e8f0;
    #         font-size: 16px;
    #         line-height: 1.6;
    #     }

    #     .main-title {
    #         color: #0ea5e9;
    #         text-align: center;
    #         margin-bottom: 30px;
    #     }

    #     .section-title {
    #         color: #38bdf8;
    #         margin-top: 30px;
    #     }

    #     .warning-box {
    #         background-color: #facc15;
    #         color: #1e293b;
    #         padding: 15px;
    #         border-left: 5px solid #f59e0b;
    #         border-radius: 8px;
    #     }
    #     </style>
    #     """,
    #     unsafe_allow_html=True,
    # )

    # Main title
    st.markdown(
        "<h1 class='main-title'>Phishing Detection with LightGBM</h1>",
        unsafe_allow_html=True,
    )

    # Image or logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/1547/1547537.png", width=200)

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
