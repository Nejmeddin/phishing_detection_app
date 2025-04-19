"""
Data exploration page for the phishing detection application.
This page allows visualization and understanding of the dataset used for
model training.
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


def generate_sample_data():
    """
    Generates synthetic data for demonstration.

    Returns:
        pd.DataFrame: DataFrame containing synthetic data
    """
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

    # Number of samples
    n_legitimate = 1000
    n_phishing = 800

    # Features for legitimate URLs
    legitimate_data = {
        "url_length": np.random.normal(50, 15, n_legitimate),
        "domain_length": np.random.normal(12, 4, n_legitimate),
        "path_length": np.random.normal(20, 10, n_legitimate),
        "query_length": np.random.normal(15, 8, n_legitimate),
        "dots_count": np.random.normal(2, 1, n_legitimate).astype(int),
        "hyphens_count": np.random.normal(0.5, 0.7, n_legitimate).astype(int),
        "underscores_count": np.random.normal(0.3, 0.6, n_legitimate).astype(int),
        "slashes_count": np.random.normal(4, 1.5, n_legitimate).astype(int),
        "is_https": np.random.binomial(1, 0.8, n_legitimate),
        "has_ip_address": np.random.binomial(1, 0.02, n_legitimate),
        "has_suspicious_tld": np.random.binomial(1, 0.05, n_legitimate),
        "subdomain_count": np.random.poisson(1, n_legitimate),
        "domain_contains_number": np.random.binomial(1, 0.15, n_legitimate),
        "has_suspicious_keywords": np.random.binomial(1, 0.1, n_legitimate),
        "domain_age": np.random.gamma(5, 1, n_legitimate),
        "CLASS_LABEL": np.zeros(n_legitimate, dtype=int),
    }

    # Features for phishing URLs
    phishing_data = {
        "url_length": np.random.normal(80, 25, n_phishing),
        "domain_length": np.random.normal(20, 7, n_phishing),
        "path_length": np.random.normal(35, 15, n_phishing),
        "query_length": np.random.normal(25, 12, n_phishing),
        "dots_count": np.random.normal(3, 1.5, n_phishing).astype(int),
        "hyphens_count": np.random.normal(1.5, 1.2, n_phishing).astype(int),
        "underscores_count": np.random.normal(1.2, 1.1, n_phishing).astype(int),
        "slashes_count": np.random.normal(5, 2, n_phishing).astype(int),
        "is_https": np.random.binomial(1, 0.4, n_phishing),
        "has_ip_address": np.random.binomial(1, 0.3, n_phishing),
        "has_suspicious_tld": np.random.binomial(1, 0.4, n_phishing),
        "subdomain_count": np.random.poisson(2, n_phishing),
        "domain_contains_number": np.random.binomial(1, 0.6, n_phishing),
        "has_suspicious_keywords": np.random.binomial(1, 0.7, n_phishing),
        "domain_age": np.random.gamma(1, 0.5, n_phishing),
        "CLASS_LABEL": np.ones(n_phishing, dtype=int),
    }

    # Creating DataFrames and merging
    df_legitimate = pd.DataFrame(legitimate_data)
    df_phishing = pd.DataFrame(phishing_data)
    df = pd.concat([df_legitimate, df_phishing], ignore_index=True)

    # Adding some derived features
    df["is_tiny_url"] = np.random.binomial(1, 0.1, len(df))
    df["ssl_valid"] = np.where(
        df["is_https"] == 1, np.random.binomial(1, 0.9, len(df)), 0
    )
    df["is_blacklisted"] = np.where(
        df["CLASS_LABEL"] == 1, np.random.binomial(1, 0.7, len(df)), 0
    )

    return df


def show_data_exploration():
    """Displays the data exploration page."""

    # Main title
    st.markdown("<h1 class='main-title'>Data Exploration</h1>", unsafe_allow_html=True)

    # Introduction
    st.markdown(
        """
    <div class='info-box'>
    <p>This section allows you to explore the dataset used to train our phishing detection model. 
    You can visualize class distribution, descriptive statistics, and correlations between different features.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Loading data
    @st.cache_data
    def load_data():
        """Loads the dataset and caches it."""
        try:
            base_path = Path(__file__).parent.parent
            data_path = os.path.join(
                base_path, "data", "raw", "Phishing_Legitimate_full.csv"
            )

            if not os.path.exists(data_path):
                # If the file doesn't exist, use simulated data
                return generate_sample_data()

            df = pd.read_csv(data_path)
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return generate_sample_data()

    # Load the data
    df = load_data()

    # Identify the target column (label)
    target_column = "CLASS_LABEL" if "CLASS_LABEL" in df.columns else "label"

    # 1. Dataset overview
    st.markdown(
        "<h2 class='section-title'>Dataset Overview</h2>",
        unsafe_allow_html=True,
    )

    # Display dataset dimensions
    st.markdown(f"**Dataset dimensions**: {df.shape[0]} rows × {df.shape[1]} columns")

    # Display first rows
    with st.expander("Data preview (first rows)"):
        st.dataframe(df.head())

    # 2. Class distribution
    st.markdown(
        "<h2 class='section-title'>Class Distribution</h2>",
        unsafe_allow_html=True,
    )

    # Calculate class distribution
    class_counts = df[target_column].value_counts()
    class_percent = df[target_column].value_counts(normalize=True) * 100

    # Create two columns for display
    col1, col2 = st.columns(2)

    with col1:
        # Distribution table
        distribution_df = pd.DataFrame(
            {
                "Class": ["Legitimate", "Phishing"],
                "Count": [class_counts.get(0, 0), class_counts.get(1, 0)],
                "Percentage": [
                    f"{class_percent.get(0, 0):.1f}%",
                    f"{class_percent.get(1, 0):.1f}%",
                ],
            }
        )
        st.table(distribution_df)

    with col2:
        # Distribution visualization
        fig = px.pie(
            names=["Legitimate", "Phishing"],
            values=[class_counts.get(0, 0), class_counts.get(1, 0)],
            color_discrete_sequence=["#6495ED", "#FF7F50"],
            title="Class Distribution",
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    # 3. Descriptive statistics
    st.markdown(
        "<h2 class='section-title'>Descriptive Statistics</h2>",
        unsafe_allow_html=True,
    )

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    # Limit to first 10 numeric columns for clarity
    if len(numeric_cols) > 10:
        selected_numeric_cols = numeric_cols[:10]
    else:
        selected_numeric_cols = numeric_cols

    # Statistics by class
    stats_legitimate = df[df[target_column] == 0][selected_numeric_cols].describe().T
    stats_phishing = df[df[target_column] == 1][selected_numeric_cols].describe().T

    # Select only certain statistics for display
    stats_legitimate = stats_legitimate[["mean", "std", "min", "max"]]
    stats_phishing = stats_phishing[["mean", "std", "min", "max"]]

    # Rename columns for clarity
    stats_legitimate.columns = [
        "Mean (Legitimate)",
        "Std Dev (Legitimate)",
        "Min (Legitimate)",
        "Max (Legitimate)",
    ]
    stats_phishing.columns = [
        "Mean (Phishing)",
        "Std Dev (Phishing)",
        "Min (Phishing)",
        "Max (Phishing)",
    ]

    # Combine statistics
    combined_stats = pd.concat([stats_legitimate, stats_phishing], axis=1)

    # Display statistics
    with st.expander("Descriptive statistics by class"):
        st.dataframe(combined_stats)

    # 4. Main features visualization
    st.markdown(
        "<h2 class='section-title'>Main Features Visualization</h2>",
        unsafe_allow_html=True,
    )

    # Select features to visualize
    key_features = [
        "url_length",
        "domain_length",
        "path_length",
        "dots_count",
        "is_https",
        "has_ip_address",
        "subdomain_count",
        "has_suspicious_keywords",
    ]

    # Filter to include only available features
    available_features = [f for f in key_features if f in df.columns]

    if not available_features:
        available_features = selected_numeric_cols[
            :4
        ]  # Use first 4 numeric columns if no key features are available

    # Feature selection interface
    selected_feature = st.selectbox(
        "Select a feature to visualize:", available_features
    )

    # Create two columns for display
    col1, col2 = st.columns(2)

    with col1:
        # Histogram of selected feature
        fig = px.histogram(
            df,
            x=selected_feature,
            color=df[target_column].map({0: "Legitimate", 1: "Phishing"}),
            color_discrete_map={"Legitimate": "#6495ED", "Phishing": "#FF7F50"},
            title=f"Distribution of '{selected_feature}' by class",
            marginal="box",
            opacity=0.7,
            barmode="overlay",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Boxplot of selected feature
        if (
            df[selected_feature].nunique() > 2
        ):  # Don't make boxplots for binary variables
            fig = px.box(
                df,
                x=df[target_column].map({0: "Legitimate", 1: "Phishing"}),
                y=selected_feature,
                color=df[target_column].map({0: "Legitimate", 1: "Phishing"}),
                color_discrete_map={"Legitimate": "#6495ED", "Phishing": "#FF7F50"},
                title=f"Boxplot of '{selected_feature}' by class",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # For binary variables, display a bar chart
            counts = (
                df.groupby([target_column, selected_feature])
                .size()
                .reset_index(name="count")
            )
            fig = px.bar(
                counts,
                x=counts[target_column].map({0: "Legitimate", 1: "Phishing"}),
                y="count",
                color=counts[selected_feature].astype(str),
                title=f"Count of '{selected_feature}' by class",
                barmode="group",
            )
            st.plotly_chart(fig, use_container_width=True)

    # 5. Correlation matrix
    st.markdown(
        "<h2 class='section-title'>Correlation Matrix</h2>", unsafe_allow_html=True
    )

    # Calculate correlation matrix
    corr_matrix = df[selected_numeric_cols + [target_column]].corr()

    # Visualize correlation matrix
    fig = px.imshow(
        corr_matrix,
        color_continuous_scale="RdBu_r",
        title="Correlations between features and target class",
        labels=dict(color="Correlation"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display correlations with target class
    st.markdown("### Correlations with target class")

    # Extract correlations with target class
    target_corr = (
        corr_matrix[target_column].drop(target_column).sort_values(ascending=False)
    )

    # Create DataFrame for display
    target_corr_df = pd.DataFrame(
        {"Feature": target_corr.index, "Correlation": target_corr.values}
    )

    # Visualize correlations
    fig = px.bar(
        target_corr_df,
        x="Correlation",
        y="Feature",
        orientation="h",
        color="Correlation",
        color_continuous_scale="RdBu_r",
        title="Correlations with target class (positive = phishing, negative = legitimate)",
    )

    st.plotly_chart(fig, use_container_width=True)

    # 6. Additional information
    st.markdown(
        "<h2 class='section-title'>Additional Information</h2>",
        unsafe_allow_html=True,
    )

    # Description of main features
    with st.expander("Description of main features"):
        st.markdown(
            """
        | Feature | Description |
        |----------------|-------------|
        | url_length | Total length of the URL |
        | domain_length | Length of the domain |
        | path_length | Length of the path in the URL |
        | query_length | Length of the query part of the URL |
        | dots_count | Number of dots in the URL |
        | hyphens_count | Number of hyphens in the URL |
        | underscores_count | Number of underscores in the URL |
        | slashes_count | Number of slashes in the URL |
        | is_https | If the URL uses HTTPS protocol (1) or HTTP (0) |
        | has_ip_address | If the URL contains an IP address instead of a domain name |
        | has_suspicious_tld | If the URL uses a suspicious domain extension |
        | subdomain_count | Number of subdomains in the URL |
        | domain_contains_number | If the domain contains numbers |
        | has_suspicious_keywords | Presence of suspicious keywords in the URL |
        | is_tiny_url | If the URL is a shortened URL |
        | domain_age | Age of the domain in years |
        | ssl_valid | If the SSL certificate is valid |
        | is_blacklisted | If the domain is present in security blacklists |
        """
        )

    # Note about the data
    st.info(
        """
    **Note**: This dataset was used to train our phishing detection model. 
    Before training, the data was preprocessed with different techniques such as normalization, 
    anomaly detection, and resampling to balance the classes.
    
    To learn more about these preprocessing steps, check the "Preprocessing" section of the application.
    """
    )
