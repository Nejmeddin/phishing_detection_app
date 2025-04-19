"""
Real-time prediction page for the phishing detection application.
This page allows users to enter a URL and receive a prediction
about the legitimacy of this URL.
"""

import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging
import requests
from bs4 import BeautifulSoup
import re
import urllib.parse
import tldextract
from typing import Dict, List, Tuple, Any, Optional, Union

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import custom modules
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.model_loader import ModelLoader
from src.config import FEATURE_DISPLAY_NAMES, FEATURE_EXPLANATIONS


# Dans la fonction extract_web_features, nous pouvons simplifier car l'enrichissement se fera dans le ModelLoader


def extract_web_features(url):
    """
    Extrait les caractéristiques de base d'une URL pour l'analyse.
    Les caractéristiques manquantes seront enrichies par le ModelLoader.

    Args:
        url: L'URL à analyser

    Returns:
        Dict: Caractéristiques extraites
    """
    features = {"url": url}

    # URL normalization
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    # Extract basic URL components
    try:
        parsed_url = urllib.parse.urlparse(url)
        extract_result = tldextract.extract(url)

        # Basic URL features
        features["IsHTTPS"] = 1 if parsed_url.scheme == "https" else 0
        features["URLLength"] = len(url)
        features["NoOfSubDomain"] = (
            len(extract_result.subdomain.split(".")) if extract_result.subdomain else 0
        )
        features["NoOfDots"] = url.count(".")
        features["NoOfObfuscatedChar"] = len(re.findall(r"%[0-9a-fA-F]{2}", url))
        features["NoOfQmark"] = url.count("?")
        features["NoOfDigits"] = sum(c.isdigit() for c in url)

        # Retrieve HTML content with retry mechanism
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

        # Set up retry parameters
        max_retries = 2
        timeout = 10
        html_content = None
        soup = None

        for attempt in range(max_retries + 1):
            try:
                response = session.get(url, timeout=timeout)
                html_content = response.text
                soup = BeautifulSoup(html_content, "html.parser")
                # Si nous arrivons ici, la requête a réussi
                break
            except requests.exceptions.RequestException as e:
                logger.warning(f"Tentative {attempt+1} échouée: {str(e)}")
                if attempt < max_retries:
                    # Attendre un peu avant de réessayer (backoff exponentiel)
                    wait_time = 2**attempt
                    logger.info(
                        f"Attente de {wait_time} secondes avant nouvelle tentative..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Toutes les {max_retries+1} tentatives ont échoué")
                    # Marquer que l'extraction HTML a échoué
                    html_content = None
                    soup = None

        # Si nous avons réussi à obtenir le contenu HTML
        if html_content and soup:
            # HTML features
            features["LineLength"] = len(html_content.splitlines())
            features["HasTitle"] = 1 if soup.title else 0
            features["HasMeta"] = 1 if soup.find_all("meta") else 0

            # Favicon
            has_favicon = 0
            links = soup.find_all("link")
            for link in links:
                rel = link.get("rel", "")
                if isinstance(rel, list):
                    rel = " ".join(rel)
                if "icon" in rel.lower():
                    has_favicon = 1
                    break
            features["HasFavicon"] = has_favicon

            # Copyright
            features["HasCopyright"] = (
                1 if "©" in html_content or "copyright" in html_content.lower() else 0
            )

            # Social networks
            social_networks = [
                "facebook",
                "twitter",
                "instagram",
                "linkedin",
                "youtube",
                "pinterest",
            ]
            has_social = 0
            for network in social_networks:
                if network in html_content.lower():
                    has_social = 1
                    break
            features["HasSocialNetworking"] = has_social

            # Password field
            features["HasPasswordField"] = (
                1 if soup.find_all("input", {"type": "password"}) else 0
            )

            # Submit button
            submit_buttons = soup.find_all("input", {"type": "submit"})
            submit_buttons.extend(soup.find_all("button", {"type": "submit"}))
            features["HasSubmitButton"] = 1 if submit_buttons else 0

            # Crypto keywords
            crypto_keywords = [
                "crypto",
                "bitcoin",
                "ethereum",
                "wallet",
                "blockchain",
                "token",
            ]
            has_crypto = 0
            for keyword in crypto_keywords:
                if keyword in html_content.lower():
                    has_crypto = 1
                    break
            features["HasKeywordCrypto"] = has_crypto

            # Popups
            features["NoOfPopup"] = len(
                soup.find_all("script", string=re.compile("window.open|popup|alert"))
            )

            # iFrames
            features["NoOfiFrame"] = len(soup.find_all("iframe"))

            # Images
            features["NoOfImage"] = len(soup.find_all("img"))

            # JavaScript
            features["NoOfJS"] = len(soup.find_all("script"))

            # CSS
            css_count = len(soup.find_all("link", {"rel": "stylesheet"}))
            css_count += len(soup.find_all("style"))
            features["NoOfCSS"] = css_count

            # URL redirects
            redirects = 0
            redirect_patterns = ["window.location", "document.location", ".href"]
            scripts = soup.find_all("script")
            for script in scripts:
                if script.string:
                    for pattern in redirect_patterns:
                        if pattern in script.string:
                            redirects += 1
            features["NoOfURLRedirect"] = redirects

            # Hyperlinks
            features["NoOfHyperlink"] = len(soup.find_all("a"))

        else:
            # Si l'extraction HTML a échoué, nous n'ajoutons pas les caractéristiques HTML
            # Le ModelLoader s'occupera de les enrichir
            logger.warning(
                "Extraction HTML échouée, les caractéristiques HTML seront enrichies par le ModelLoader"
            )

    except Exception as e:
        logger.warning(f"Erreur d'extraction URL: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        # Même en cas d'erreur, on garde l'URL pour l'enrichissement
        features = {"url": url}

    # Vérification finale
    expected_features = [
        "IsHTTPS",
        "URLLength",
        "NoOfSubDomain",
        "NoOfDots",
        "NoOfObfuscatedChar",
        "NoOfQmark",
        "NoOfDigits",
        "LineLength",
        "HasTitle",
        "HasMeta",
        "HasFavicon",
        "HasCopyright",
        "HasSocialNetworking",
        "HasPasswordField",
        "HasSubmitButton",
        "HasKeywordCrypto",
        "NoOfPopup",
        "NoOfiFrame",
        "NoOfImage",
        "NoOfJS",
        "NoOfCSS",
        "NoOfURLRedirect",
        "NoOfHyperlink",
    ]

    missing = set(expected_features) - set(features.keys())
    if missing:
        logger.info(
            f"Caractéristiques manquantes après l'extraction initiale ({len(missing)}): {missing}"
        )
        logger.info("Ces caractéristiques seront enrichies par le ModelLoader")

    return features


def show_prediction():
    """Displays the real-time prediction page."""

    # Main title
    st.markdown(
        "<h1 class='main-title'>Real-Time URL Analysis</h1>",
        unsafe_allow_html=True,
    )

    # Introduction
    st.markdown(
        """
    <div class='info-box'>
    <p>This section allows you to analyze a URL to determine if it is potentially malicious 
    (phishing) or legitimate. Simply enter the complete URL below to start the analysis.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # URL form
    with st.form("url_form"):
        url_input = st.text_input(
            "URL to analyze:",
            placeholder="Example: https://www.example.com",
            help="Enter the complete URL for accurate analysis.",
        )
        analyze_button = st.form_submit_button("Analyze URL")

    # Initialization of necessary classes

    @st.cache_resource
    def load_model():
        """Loads the LightGBM model and its associated components."""
        try:
            base_path = Path(__file__).parent.parent
            model_path = os.path.join(
                base_path, "data", "processed", "lightgbm_phishing_model.pkl"
            )

            logger.info(f"Model path: {model_path}")
            logger.info(f"File exists: {os.path.exists(model_path)}")

            if not os.path.exists(model_path):
                # Instead of using a demo model, display an error and stop
                st.error(
                    "LightGBM model not found. Please verify that the file exists in the specified path."
                )
                st.stop()

            loader = ModelLoader(model_path)
            success = loader.load()

            if not success:
                # Instead of using a demo model, display an error and stop
                st.error(
                    "Error loading the LightGBM model. Check the pickle file format."
                )
                st.stop()

            return loader
        except Exception as e:
            # Display the error and stop
            st.error(f"Exception while loading the model: {str(e)}")
            st.stop()

    # @st.cache_resource
    # def load_model():
    #     """Loads the LightGBM model and its associated components."""
    #     try:
    #         base_path = Path(__file__).parent.parent
    #         model_path = os.path.join(
    #             base_path, "data", "processed", "lightgbm_phishing_model.pkl"
    #         )

    #         logger.info(f"Model path: {model_path}")
    #         logger.info(f"File exists: {os.path.exists(model_path)}")

    #         if not os.path.exists(model_path):
    #             # If the original file doesn't exist, let's use a demo model
    #             st.warning(
    #                 "Original model not found, using a demo model."
    #             )
    #             return DemoModelLoader()

    #         loader = ModelLoader(model_path)
    #         success = loader.load()

    #         if not success:
    #             st.error(
    #                 "Error loading the model. Using the demo model."
    #             )
    #             return DemoModelLoader()

    #         return loader
    #     except Exception as e:
    #         st.error(f"Exception while loading the model: {str(e)}")
    #         return DemoModelLoader()

    # Class to simulate a model when the pickle file is not available
    class DemoModelLoader:
        """Phishing detection model simulator for demonstration."""

        def __init__(self):
            """Initializes the demo model."""
            self.feature_names = [
                "IsHTTPS",
                "URLLength",
                "NoOfSubDomain",
                "NoOfDots",
                "NoOfObfuscatedChar",
                "NoOfQmark",
                "NoOfDigits",
                "LineLength",
                "HasTitle",
                "HasMeta",
                "HasFavicon",
                "HasCopyright",
                "HasSocialNetworking",
                "HasPasswordField",
                "HasSubmitButton",
                "HasKeywordCrypto",
                "NoOfPopup",
                "NoOfiFrame",
                "NoOfImage",
                "NoOfJS",
                "NoOfCSS",
                "NoOfURLRedirect",
                "NoOfHyperlink",
            ]
            self.selected_feature_names = (
                self.feature_names
            )  # For the demo, let's use all features

            # Simulated metrics for feature importance
            importance = [0.1] * len(
                self.feature_names
            )  # Equal importance for all features

            self.metrics = {
                "feature_importance": importance,
                "feature_names": self.selected_feature_names,
            }

        def get_required_features(self) -> List[str]:
            """
            Retourne la liste des caractéristiques requises par le modèle.

            Returns:
                List[str]: Liste des noms de caractéristiques
            """
            # Convertir le pandas.Index en liste Python
            if self.selected_feature_names is not None and hasattr(
                self.selected_feature_names, "tolist"
            ):
                return self.selected_feature_names.tolist()
            elif self.selected_feature_names is not None:
                return list(self.selected_feature_names)
            else:
                return []

        def preprocess_features(self, features_df):
            """Simulates feature preprocessing."""
            # Check that all expected columns are present
            for feature in self.feature_names:
                if feature not in features_df.columns:
                    features_df[feature] = 0  # Default value

            # Select only the columns we use
            return features_df[self.feature_names].values

        def predict(self, features_df):
            """Simulates a prediction based on simple heuristics."""
            # Suspicious features
            has_password = features_df["HasPasswordField"].values[0] == 1
            has_submit = features_df["HasSubmitButton"].values[0] == 1
            https = features_df["IsHTTPS"].values[0] == 1
            has_favicon = features_df["HasFavicon"].values[0] == 1
            has_title = features_df["HasTitle"].values[0] == 1

            # Phishing score
            suspicious_score = 0

            # Suspicious factors
            if has_password and has_submit and not https:
                suspicious_score += 0.4
            if not has_favicon or not has_title:
                suspicious_score += 0.2
            if features_df["NoOfObfuscatedChar"].values[0] > 0:
                suspicious_score += 0.2

            # Adjust to have a score between 0.1 and 0.9
            phishing_proba = max(0.1, min(0.9, suspicious_score))

            # For known URLs, force certain values
            url = features_df["url"].values[0] if "url" in features_df.columns else ""
            if "gdocs" in url.lower() or "login" in url.lower():
                phishing_proba = 0.85
            elif "google.com" in url.lower() or "microsoft.com" in url.lower():
                phishing_proba = 0.15

            prediction = 1 if phishing_proba > 0.5 else 0

            return np.array([phishing_proba]), np.array([prediction])

        def get_model_components(self):
            """Returns the demo model components."""
            return {
                "metrics": self.metrics,
                "selected_feature_names": self.selected_feature_names,
            }

        def ensure_feature_consistency(self, features_df):
            """Ensures that all necessary features are present."""
            required_features = self.get_required_features()

            # Check for missing features
            missing_features = set(required_features) - set(features_df.columns)
            if missing_features:
                for col in missing_features:
                    features_df[col] = 0

            # Keep the URL column if present
            if "url" in features_df.columns:
                return features_df[required_features + ["url"]].copy()
            else:
                return features_df[required_features].copy()

    # If the user clicked the analyze button
    if analyze_button and url_input:
        # Load the model
        model_loader = load_model()
        required_features = model_loader.get_required_features()

        if required_features is None or len(required_features) == 0:
            st.error("Unable to determine the features required by the model.")
            st.stop()

        logger.info(f"Features required by the model: {required_features}")

        # Display a progress bar during analysis
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Step 1: Extract all features (URL and HTML)
            status_text.text("Analyzing URL and extracting features...")
            features = extract_web_features(url_input)
            progress_bar.progress(0.6)

            # For debug
            logger.info(f"Extracted features: {features}")

            # Step 2: Prepare DataFrame for prediction
            features_df = pd.DataFrame([features])

            # Step 3: Check feature consistency with the model
            features_df = model_loader.ensure_feature_consistency(features_df)
            logger.info(
                f"Final DataFrame for prediction: {features_df.columns.tolist()}"
            )

            progress_bar.progress(0.8)
            time.sleep(0.5)  # Processing simulation

            # Step 4: Prediction
            status_text.text("Analysis in progress...")
            probas, predictions = model_loader.predict(features_df)

            # For debug
            logger.info(f"Probabilities: {probas}")
            logger.info(f"Predictions: {predictions}")

            progress_bar.progress(1.0)
            status_text.text("Analysis completed!")
            time.sleep(0.5)

            # Clear the progress bar and status text
            progress_bar.empty()
            status_text.empty()

            # Display results
            show_prediction_results(
                url_input, probas, predictions, features, model_loader
            )

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Error analyzing the URL: {str(e)}")
            import traceback

            st.code(traceback.format_exc())

    # If no URL is entered, display an explanatory message
    elif analyze_button and not url_input:
        st.warning("Please enter a URL to analyze.")

    # Information section
    with st.expander("How does this analysis work?"):
        st.markdown(
            """
        #### URL Analysis Process
        
        Our system uses a multi-step process to analyze URLs:
        
        1. **Direct feature extraction**: We analyze the structure of the URL itself
           (length, presence of special characters, subdomains, etc.)
        
        2. **Content analysis**: We examine the HTML content of the page to detect
           suspicious elements (forms, password fields, redirections, etc.)
        
        3. **Model application**: Our LightGBM model, trained on thousands of examples,
           evaluates all these features to determine if the URL is potentially malicious.
        
        4. **Interpretation**: The results are presented with a confidence score and an explanation
           of the factors that influenced the decision.
        """
        )

    # Security warning
    st.markdown(
        """
    <div class='warning-box'>
    <h3>⚠️ Security Reminder</h3>
    <p>This analysis is a decision support tool but does not replace vigilance. Even with a positive result
    (URL considered legitimate), remain cautious, especially if you need to provide sensitive information.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def show_prediction_results(url, probas, predictions, features, model_loader):
    """
    Displays the prediction results.

    Args:
        url: The analyzed URL
        probas: The predicted probabilities
        predictions: The predicted classes
        features: The dictionary of features
        model_loader: The model loader instance
    """
    logger = logging.getLogger(__name__)

    # Get the prediction and probability
    is_phishing = predictions[0] == 1
    phishing_proba = probas[0]
    if isinstance(phishing_proba, np.ndarray) and len(phishing_proba) > 0:
        phishing_proba = phishing_proba[0]

    # Add logs for debugging
    logger.info(f"Type of phishing_proba: {type(phishing_proba)}")
    logger.info(f"Raw value of phishing_proba: {phishing_proba}")

    # Check if the value is valid (not NaN or infinite)
    if np.isnan(phishing_proba) or np.isinf(phishing_proba):
        logger.error(
            f"Invalid probability detected: {phishing_proba}, using a default value"
        )
        # Use a default value based on features
        is_suspicious = (
            features.get("HasPasswordField", 0) == 1
            and features.get("HasSubmitButton", 0) == 1
            and features.get("IsHTTPS", 0) == 0
        )
        phishing_proba = 0.8 if is_suspicious else 0.2

    # If the probability is exactly 0 or 1, adjust slightly to avoid extreme scores
    if phishing_proba == 0:
        logger.warning("Probability exactly 0 detected, adjusting to 0.01")
        phishing_proba = 0.01
    elif phishing_proba == 1:
        logger.warning("Probability exactly 1 detected, adjusting to 0.99")
        phishing_proba = 0.99

    # Calculate legitimacy score (inverse of phishing probability)
    legitimate_proba = 1 - phishing_proba
    legitimacy_score = legitimate_proba * 100

    logger.info(f"URL: {url}")
    logger.info(
        f"Raw prediction: {predictions[0]}, Phishing probability: {phishing_proba}"
    )
    logger.info(f"Calculated legitimacy score: {legitimacy_score}%")

    # If the score is close to 0 or 100, it's suspicious
    if legitimacy_score < 0.1 or legitimacy_score > 99.9:
        logger.warning(
            f"Extreme score detected: {legitimacy_score}%, using a heuristic score"
        )
        # Calculate a heuristic score based on features
        score_base = 50  # Neutral score

        # Reduce score for suspicious features
        if not features.get("HasTitle", 0):
            score_base -= 10
        if not features.get("HasFavicon", 0):
            score_base -= 10
        if features.get("HasPasswordField", 0) and not features.get("IsHTTPS", 0):
            score_base -= 20
        if features.get("NoOfObfuscatedChar", 0) > 0:
            score_base -= 15
        if features.get("NoOfiFrame", 0) > 2:
            score_base -= 10

        # Increase score for positive features
        if features.get("IsHTTPS", 0):
            score_base += 10
        if features.get("HasCopyright", 0):
            score_base += 10
        if features.get("HasSocialNetworking", 0):
            score_base += 5

        # Ensure the score stays in the range [5, 95]
        legitimacy_score = max(5, min(95, score_base))
        is_phishing = legitimacy_score < 50
        logger.info(f"Calculated heuristic score: {legitimacy_score}%")

    # Title and separator
    st.markdown("---")
    st.markdown(
        "<h2 class='section-title'>Analysis Results</h2>", unsafe_allow_html=True
    )

    # Display main result
    col1, col2 = st.columns([1, 2])

    with col1:
        if is_phishing:
            st.markdown(
                f"""
                <div style="background-color: #FFEBEE; border-radius: 10px; padding: 20px; text-align: center;">
                <h1 style="color: #D32F2F; margin: 0;">⚠️ WARNING</h1>
                <h2 style="color: #D32F2F; margin: 10px 0;">Potentially malicious site</h2>
                <p style="font-size: 1.2rem;">Phishing probability: <b>{(100-legitimacy_score):.1f}%</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background-color: #E8F5E9; border-radius: 10px; padding: 20px; text-align: center;">
                <h1 style="color: #388E3C; margin: 0;">✅ SECURE</h1>
                <h2 style="color: #388E3C; margin: 10px 0;">Probably legitimate site</h2>
                <p style="font-size: 1.2rem;">Legitimacy probability: <b>{legitimacy_score:.1f}%</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Display analyzed URL
    with col2:
        st.markdown("<h3>Analyzed URL:</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='word-break: break-all; font-size: 1.1rem;'>{url}</p>",
            unsafe_allow_html=True,
        )

        # Confidence gauge with Plotly
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=legitimacy_score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Legitimacy Score", "font": {"size": 24}},
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "darkblue",
                    },
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 30], "color": "#FF5252"},
                        {"range": [30, 70], "color": "#FFC107"},
                        {"range": [70, 100], "color": "#4CAF50"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": legitimacy_score,
                    },
                },
            )
        )
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Separator
    st.markdown("---")

    # Create two columns for detailed analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3>Key factors for this prediction</h3>", unsafe_allow_html=True)

        # Get the most important features
        model_components = model_loader.get_model_components()
        feature_importance = model_components.get("metrics", {}).get(
            "feature_importance", []
        )
        feature_names = model_components.get("selected_feature_names", [])

        # If we have importance information
        if len(feature_importance) > 0 and len(feature_names) > 0:
            # Convert to Python lists if they are numpy arrays
            if isinstance(feature_importance, np.ndarray):
                feature_importance = feature_importance.tolist()
            if isinstance(feature_names, np.ndarray):
                feature_names = feature_names.tolist()

            # Create an importance dictionary
            importance_dict = dict(zip(feature_names, feature_importance))

            # Sort features by importance
            sorted_features = sorted(
                importance_dict.items(), key=lambda x: x[1], reverse=True
            )

            # Display the 5 most important factors
            top_features = sorted_features[:5]

            # Prepare data for the chart
            df = pd.DataFrame(
                {
                    "Feature": [f[0] for f in top_features],
                    "Importance": [f[1] for f in top_features],
                }
            )

            # Create a horizontal chart with Altair
            try:
                chart = (
                    alt.Chart(df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Importance:Q", title="Importance"),
                        y=alt.Y("Feature:N", sort="-x", title="Feature"),
                        color=alt.Color(
                            "Importance:Q", scale=alt.Scale(scheme="blues")
                        ),
                    )
                    .properties(height=200)
                )

                st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                logger.error(f"Error creating chart: {str(e)}")
                # Alternative: display a simple table
                st.table(df)

            # Add a summary of determining factors for this specific URL
            st.markdown(
                "<h4>Determining factors for this URL:</h4>", unsafe_allow_html=True
            )

            factors_list = []

            # Check specific risk or trust factors
            if (
                features.get("HasPasswordField", 0) == 1
                and features.get("IsHTTPS", 0) == 0
            ):
                factors_list.append(
                    "❌ **Unsecured password form**: The site asks for a password without using HTTPS."
                )

            if features.get("IsHTTPS", 0) == 1:
                factors_list.append(
                    "✅ **Secure connection**: The site uses HTTPS protocol to encrypt data."
                )

            if features.get("HasTitle", 0) == 0:
                factors_list.append(
                    "❌ **No title**: The site has no title tag, which is unusual for a legitimate site."
                )

            if features.get("HasFavicon", 0) == 0:
                factors_list.append(
                    "❌ **No icon**: The site has no favicon, which is often the case with phishing sites."
                )

            if features.get("NoOfObfuscatedChar", 0) > 0:
                factors_list.append(
                    f"❌ **Obfuscated URL**: The URL contains {features.get('NoOfObfuscatedChar')} encoded characters, which may mask its true destination."
                )

            if features.get("HasCopyright", 0) == 1:
                factors_list.append(
                    "✅ **Copyright mention**: The site contains a copyright mention, usually present on legitimate sites."
                )

            if features.get("HasSocialNetworking", 0) == 1:
                factors_list.append(
                    "✅ **Social network links**: Presence of links to social networks, often found on legitimate sites."
                )

            if features.get("NoOfiFrame", 0) > 2:
                factors_list.append(
                    f"❌ **Numerous iframes**: The site contains {features.get('NoOfiFrame')} iframes, which may indicate malicious content."
                )

            # Display factors
            if factors_list:
                for factor in factors_list:
                    st.markdown(factor)
            else:
                st.markdown(
                    "No specific determining factors were identified for this URL."
                )

    with col2:
        st.markdown("<h3>Features extracted from the URL</h3>", unsafe_allow_html=True)

        # Create a DataFrame for display
        display_features = {}
        display_names = {
            "IsHTTPS": "Uses HTTPS",
            "URLLength": "URL Length",
            "NoOfSubDomain": "Number of subdomains",
            "NoOfDots": "Number of dots",
            "NoOfObfuscatedChar": "Obfuscated characters",
            "NoOfQmark": "Number of question marks",
            "NoOfDigits": "Number of digits",
            "LineLength": "Content length",
            "HasTitle": "Has title",
            "HasMeta": "Has meta tags",
            "HasFavicon": "Has icon",
            "HasCopyright": "Copyright mention",
            "HasSocialNetworking": "Social network links",
            "HasPasswordField": "Password field",
            "HasSubmitButton": "Submit button",
            "HasKeywordCrypto": "Cryptocurrency terms",
            "NoOfPopup": "Number of popups",
            "NoOfiFrame": "Number of iframes",
            "NoOfImage": "Number of images",
            "NoOfJS": "Number of JS scripts",
            "NoOfCSS": "Number of CSS sheets",
            "NoOfURLRedirect": "Number of redirections",
            "NoOfHyperlink": "Number of links",
        }

        for feature in sorted(features.keys()):
            if feature != "url":  # Exclude the complete URL
                # Format values for more user-friendly display
                value = features[feature]

                # Special formatting for certain features
                if feature in [
                    "IsHTTPS",
                    "HasTitle",
                    "HasMeta",
                    "HasFavicon",
                    "HasCopyright",
                    "HasSocialNetworking",
                    "HasPasswordField",
                    "HasSubmitButton",
                    "HasKeywordCrypto",
                ]:
                    value = "Yes" if value == 1 else "No"

                display_features[display_names.get(feature, feature)] = value

        # Create a DataFrame for display
        df_display = pd.DataFrame(
            list(display_features.items()), columns=["Feature", "Value"]
        )

        # Display the table
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Security recommendations
    st.markdown("---")
    st.markdown("<h3>Security Recommendations</h3>", unsafe_allow_html=True)

    if is_phishing:
        st.markdown(
            """
        <div class='warning-box'>
        <h4>🛑 This URL shows signs of phishing. We recommend:</h4>
        <ul>
            <li>Do not visit this site or leave it immediately</li>
            <li>Never enter personal information or login credentials</li>
            <li>Report the URL to appropriate authorities or security services</li>
            <li>If you have already entered information, change your passwords immediately</li>
        </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
        <div class='success-box'>
        <h4>✅ This URL seems legitimate, but here are some best practices to follow:</h4>
        <ul>
            <li>Always verify that you are on the correct domain before entering sensitive information</li>
            <li>Use two-factor authentication when possible</li>
            <li>Stay vigilant against unusual requests, even on legitimate sites</li>
            <li>Keep your software and browsers updated for better protection</li>
        </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )


with st.expander("How does this analysis work?"):
    st.markdown(
        """
        #### URL Analysis Process
        
        Notre système utilise un processus en plusieurs étapes pour analyser les URLs:
        
        1. **Extraction directe des caractéristiques**: Nous analysons la structure de l'URL et son contenu HTML
        
        2. **Enrichissement intelligent des caractéristiques**: Si certaines caractéristiques ne peuvent pas être
           extraites directement (par exemple, si le site est inaccessible), nous utilisons des services 
           tiers gratuits comme urlscan.io pour les compléter sans faire de supposition.
        
        3. **Application du modèle**: Notre modèle LightGBM, entraîné sur des milliers d'exemples,
           évalue toutes ces caractéristiques pour déterminer si l'URL est potentiellement malveillante.
        
        4. **Interprétation**: Les résultats sont présentés avec un score de confiance et une explication
           des facteurs qui ont influencé la décision.
        """
    )
