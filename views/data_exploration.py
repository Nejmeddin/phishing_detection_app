"""
Data exploration page for the phishing detection application.
This page allows visualization and understanding of the dataset used for
model training.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import PHISHING_DATASET_PATH


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

    @st.cache_data
    def load_data():
        """Load the training dataset.

        The page reports on the real dataset or not at all: silently swapping in
        generated data would make every statistic below a fabrication.
        """
        if not PHISHING_DATASET_PATH.exists():
            st.error(
                f"Dataset not found at `{PHISHING_DATASET_PATH}`. "
                "Restore it to explore the data."
            )
            st.stop()

        try:
            return pd.read_csv(PHISHING_DATASET_PATH)
        except (OSError, pd.errors.ParserError) as exc:
            st.error(f"Could not read the dataset: {exc}")
            st.stop()

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
        labels={"color": "Correlation"},
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
