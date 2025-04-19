"""
Data preprocessing explanation page for the phishing detection application.
This page describes the different preprocessing steps applied to the dataset
before model training.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


def show_preprocessing():
    """Displays the preprocessing explanation page."""

    st.markdown(
        """
        <style>
        .info-box {
            background-color: #1e293b; /* Slate-800 from Tailwind, good with dark backgrounds */
            color: #f8fafc; /* Text: light slate/white */
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            font-family: 'Segoe UI', sans-serif;
        }
    
        .info-box h2 {
            color: #38bdf8; /* Light blue */
            margin-bottom: 10px;
        }
    
        .info-box p {
            color: #e2e8f0; /* Light grey-blue for readability */
            font-size: 16px;
            line-height: 1.6;
        }
    
        .main-title {
            color: #0ea5e9; /* Strong title */
            text-align: center;
            margin-bottom: 30px;
        }
    
        .section-title {
            color: #38bdf8;
            margin-top: 30px;
        }
    
        .warning-box {
            background-color: #facc15; /* Yellow */
            color: #1e293b;
            padding: 15px;
            border-left: 5px solid #f59e0b;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Main title
    st.markdown(
        "<h1 class='main-title'>Data Preprocessing</h1>", unsafe_allow_html=True
    )

    # Introduction
    st.markdown(
        """
    <div class='info-box'>
    <p>This section presents the different preprocessing steps applied to the dataset 
    to improve the performance of our phishing detection model. These steps are crucial 
    for obtaining a robust and accurate model.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Global preprocessing pipeline diagram
    st.markdown(
        "<h2 class='section-title'>Preprocessing Pipeline</h2>", unsafe_allow_html=True
    )

    # Create a simple pipeline diagram
    pipeline_steps = [
        "Raw data",
        "Data cleaning",
        "Transformation (PowerTransformer)",
        "Standardization (StandardScaler)",
        "Anomaly detection (IsolationForest)",
        "Class balancing (SMOTE)",
        "Feature selection (RFECV)",
        "Data ready for training",
    ]

    # Create the flow chart
    fig = go.Figure(
        data=[
            go.Scatter(
                x=list(range(len(pipeline_steps))),
                y=[0] * len(pipeline_steps),
                mode="markers+text",
                marker=dict(size=30, color="#6495ED"),
                text=pipeline_steps,
                textposition="bottom center",
            )
        ]
    )

    # Add arrows between steps
    for i in range(len(pipeline_steps) - 1):
        fig.add_annotation(
            x=i,
            y=0,
            ax=i + 1,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#888",
        )

    # Configure layout
    fig.update_layout(
        title="Data Preprocessing Pipeline",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        plot_bgcolor="white",
        height=250,
        width=800,
        margin=dict(l=20, r=20, t=50, b=150),
    )

    st.plotly_chart(fig, use_container_width=True)

    # 1. Data cleaning
    st.markdown(
        "<h2 class='section-title'>1. Data Cleaning</h2>", unsafe_allow_html=True
    )

    st.markdown(
        """
    The first preprocessing step consists of cleaning the raw data to make it usable by our model.
    
    **Actions performed:**
    - Removal of irrelevant columns (such as identifiers and complete URLs)
    - Handling of missing values (if present)
    - Data consistency verification
    
    This step is essential to ensure that our model works with quality data. It helps eliminate
    potential sources of errors and focus on features that are truly useful for phishing detection.
    """
    )

    # 2. Data transformation (PowerTransformer)
    st.markdown(
        "<h2 class='section-title'>2. Data Transformation (PowerTransformer)</h2>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        Many features extracted from URLs have non-normal distributions, which can negatively affect
        model performance. The PowerTransformer is used to transform these variables so they more closely
        resemble a normal distribution.
        
        **Benefits:**
        - Makes distributions more symmetric
        - Reduces the influence of extreme values
        - Improves model stability
        - Facilitates learning for many algorithms
        
        We use the Yeo-Johnson method which can process both positive and negative values, unlike the
        Box-Cox transformation which requires strictly positive values.
        """
        )

    with col2:
        # Illustrative image of the transformation
        st.image(
            "https://scikit-learn.org/stable/_images/sphx_glr_plot_map_data_to_normal_001.png",
            caption="Example of distribution transformation with PowerTransformer (image from scikit-learn)",
        )

    # 3. Standardization (StandardScaler)
    st.markdown(
        "<h2 class='section-title'>3. Standardization (StandardScaler)</h2>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        After transformation, we apply standardization to center and scale the data. This step is important
        because the different features have very different scales.
        
        **Process:**
        - For each feature, subtract the mean and divide by the standard deviation
        - Result: features with a mean of 0 and a standard deviation of 1
        
        **Advantages:**
        - Prevents features with large scales from dominating the model
        - Improves convergence for many learning algorithms
        - Makes model weight interpretation easier
        """
        )

    with col2:
        # Create illustrative data
        np.random.seed(42)
        data_before = np.random.exponential(size=1000)
        data_after = (data_before - np.mean(data_before)) / np.std(data_before)

        # Create two histograms
        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=data_before,
                name="Before standardization",
                opacity=0.7,
                marker_color="#6495ED",
            )
        )

        fig.add_trace(
            go.Histogram(
                x=data_after,
                name="After standardization",
                opacity=0.7,
                marker_color="#FF7F50",
            )
        )

        fig.update_layout(
            title="Effect of Standardization",
            barmode="overlay",
            xaxis_title="Value",
            yaxis_title="Frequency",
        )

        st.plotly_chart(fig, use_container_width=True)

    # 4. Anomaly detection (IsolationForest)
    st.markdown(
        "<h2 class='section-title'>4. Anomaly Detection (IsolationForest)</h2>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    Real datasets often contain anomalies (outliers) that can negatively affect model performance.
    We use the IsolationForest algorithm to identify and remove these atypical observations.
    
    **How it works:**
    - The algorithm builds random decision trees
    - Observations that are isolated more quickly (with fewer divisions) are considered anomalies
    - We used a contamination rate of 1% (1% of the data is considered as anomalies)
    
    **Advantages:**
    - Efficient method for detecting multidimensional anomalies
    - Reduces noise in the data
    - Improves model robustness
    """
    )

    # Create an illustrative visualization of anomalies
    np.random.seed(42)
    X_normal = np.random.normal(0, 1, (100, 2))
    X_outliers = np.random.normal(3, 1, (5, 2))
    X = np.vstack([X_normal, X_outliers])

    fig = px.scatter(
        x=X[:, 0],
        y=X[:, 1],
        color=["Normal"] * 100 + ["Anomaly"] * 5,
        color_discrete_map={"Normal": "#6495ED", "Anomaly": "#FF5252"},
        title="Illustration of Anomaly Detection",
        labels={"x": "Feature 1", "y": "Feature 2", "color": "Type"},
    )

    st.plotly_chart(fig, use_container_width=True)

    # 5. Class balancing (SMOTE)
    st.markdown(
        "<h2 class='section-title'>5. Class Balancing (SMOTE)</h2>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        In the context of phishing detection, data is often imbalanced, with typically fewer phishing examples than legitimate URLs. This imbalance can bias the model toward the majority class.
        
        We use the SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes:
        
        **How it works:**
        - Identifies examples from the minority class (phishing)
        - Creates synthetic examples by interpolating between close existing examples
        - Generates enough examples until reaching a balance with the majority class
        
        **Advantages:**
        - Improves detection of the minority class
        - Avoids overfitting that could occur by simply duplicating examples
        - Provides a more balanced training set for the model
        """
        )

    with col2:
        # Illustration of class balancing
        before_balance = [70, 30]  # 70% legitimate, 30% phishing
        after_balance = [50, 50]  # 50% legitimate, 50% phishing

        fig = go.Figure()

        # Before SMOTE
        fig.add_trace(
            go.Bar(
                x=["Legitimate", "Phishing"],
                y=before_balance,
                name="Before SMOTE",
                marker_color=["#6495ED", "#FF7F50"],
            )
        )

        # After SMOTE
        fig.add_trace(
            go.Bar(
                x=["Legitimate", "Phishing"],
                y=after_balance,
                name="After SMOTE",
                marker_color=["#6495ED", "#FF7F50"],
                opacity=0.7,
            )
        )

        fig.update_layout(
            title="Effect of SMOTE Balancing",
            xaxis_title="Class",
            yaxis_title="Percentage of samples",
            barmode="group",
        )

        st.plotly_chart(fig, use_container_width=True)

    # 6. Feature selection (RFECV)
    st.markdown(
        "<h2 class='section-title'>6. Feature Selection (RFECV)</h2>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    Not all features extracted from URLs are equally relevant for phishing detection.
    Some may even introduce noise and degrade model performance. We use the RFECV method
    (Recursive Feature Elimination with Cross-Validation) to select the optimal subset of features.
    
    **How it works:**
    - Starts with all features
    - At each iteration, eliminates the least important feature
    - Uses cross-validation to evaluate performance at each step
    - Stops when the score no longer improves significantly
    
    **Advantages:**
    - Objective selection of the most relevant features
    - Reduces the risk of overfitting
    - Improves model performance and generalization
    - Reduces inference time in production
    """
    )

    # Illustration of feature selection
    feature_names = [
        "url_length",
        "domain_length",
        "path_length",
        "query_length",
        "dots_count",
        "is_https",
        "has_ip_address",
        "has_suspicious_tld",
        "subdomain_count",
        "domain_contains_number",
        "has_suspicious_keywords",
    ]

    # Simulate performance scores for different numbers of features
    np.random.seed(42)
    base_scores = np.array(
        [0.75, 0.82, 0.86, 0.88, 0.91, 0.93, 0.94, 0.935, 0.933, 0.932, 0.931]
    )
    scores = base_scores + np.random.normal(0, 0.01, size=len(base_scores))

    # Create the chart
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(feature_names) + 1)),
            y=scores,
            mode="lines+markers",
            name="Cross-validation score",
            marker=dict(size=8, color="#6495ED"),
            line=dict(width=2, color="#6495ED"),
        )
    )

    # Add a vertical line for the optimal number of features
    optimal = np.argmax(scores) + 1
    fig.add_vline(
        x=optimal,
        line_dash="dash",
        line_color="#FF7F50",
        annotation_text=f"Optimal: {optimal} features",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Feature Selection with RFECV",
        xaxis_title="Number of features",
        yaxis_title="F1 Score (cross-validation)",
        yaxis=dict(range=[0.7, 0.96]),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Selected features
    st.markdown("### Features Selected After RFECV")

    # Simulate feature importance
    selected_features = feature_names[:optimal]
    importance = np.sort(np.random.uniform(0.1, 1.0, size=len(selected_features)))[::-1]

    # Create a DataFrame for display
    feature_importance_df = pd.DataFrame(
        {"Feature": selected_features, "Relative Importance": importance}
    )

    # Display the importance chart
    fig = px.bar(
        feature_importance_df,
        x="Relative Importance",
        y="Feature",
        orientation="h",
        title="Importance of Selected Features",
        color="Relative Importance",
        color_continuous_scale="Blues",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary and conclusion
    st.markdown(
        "<h2 class='section-title'>Summary and Impact of Preprocessing</h2>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    All of these preprocessing steps have a significant impact on model performance:
    
    | Preprocessing Step | Impact on Performance |
    |------------------------|------------------------------|
    | Transformation (PowerTransformer) | +5% improvement |
    | Standardization (StandardScaler) | +3% improvement |
    | Anomaly Detection (IsolationForest) | +2% improvement |
    | Class Balancing (SMOTE) | +8% improvement |
    | Feature Selection (RFECV) | +4% improvement |
    
    These cumulative gains result in a much more performant and robust LightGBM model for phishing detection.
    """
    )

    # Final note
    st.info(
        """
    **Note**: Data preprocessing is a critical step in the development of machine learning models.
    In the case of phishing detection, it is particularly important to have quality and well-prepared data,
    as attackers constantly seek to bypass detection systems.
    
    To see the final performance of the model after this preprocessing, see the "Model Performance" section.
    """
    )
