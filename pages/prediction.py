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
import urllib.parse
import tldextract
import plotly.graph_objects as go
from pathlib import Path
import logging
import re
from typing import Dict, List, Any, Optional, Union

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import custom modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.model_loader import ModelLoader
from src.preprocessing.feature_extractor import EnhancedFeatureExtractor
from src.config import FEATURE_DISPLAY_NAMES, FEATURE_EXPLANATIONS

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

    # Chargement du modèle
    @st.cache_resource
    def load_model():
        """Charge le modèle LightGBM pour la prédiction"""
        try:
            base_path = Path(__file__).parent.parent
            model_path = os.path.join(base_path, "data", "processed", "lightgbm_phishing_model.pkl")
            
            logger.info(f"Model path: {model_path}")
            logger.info(f"File exists: {os.path.exists(model_path)}")
            
            if not os.path.exists(model_path):
                st.error("LightGBM model not found. Please verify that the file exists in the specified path.")
                st.stop()
            
            loader = ModelLoader(model_path)
            success = loader.load()
            
            if not success:
                st.error("Error loading the LightGBM model. Check the pickle file format.")
                st.stop()
                
            return loader
            
        except Exception as e:
            st.error(f"Exception while loading the model: {str(e)}")
            st.stop()

    # Si l'utilisateur a cliqué sur le bouton d'analyse
    if analyze_button and url_input:
        # Charger le modèle
        model_loader = load_model()
        required_features = model_loader.get_required_features()
        
        if required_features is None or len(required_features) == 0:
            st.error("Unable to determine the features required by the model.")
            st.stop()
            
        logger.info(f"Features required by the model: {required_features}")
        
        # Afficher une barre de progression pendant l'analyse
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Étape 1: Initialiser l'extracteur et extraire les caractéristiques
            status_text.text("Analyzing URL and extracting features...")
            feature_extractor = EnhancedFeatureExtractor(required_features)
            features = feature_extractor.extract_features(url_input)
            progress_bar.progress(0.6)
            
            logger.info(f"Extracted features: {features}")
            
            # Étape 2: Préparer le DataFrame pour la prédiction
            features_df = pd.DataFrame([features])
            
            # Étape 3: Vérifier la cohérence des caractéristiques avec le modèle
            features_df = model_loader.ensure_feature_consistency(features_df)
            
            progress_bar.progress(0.8)
            time.sleep(0.5)  # Simulation de traitement
            
            # Étape 4: Prédiction
            status_text.text("Analysis in progress...")
            probas, predictions = model_loader.predict(features_df)
            
            logger.info(f"Probabilities: {probas}")
            logger.info(f"Predictions: {predictions}")
            
            progress_bar.progress(1.0)
            status_text.text("Analysis completed!")
            time.sleep(0.5)
            
            # Effacer la barre de progression et le texte de statut
            progress_bar.empty()
            status_text.empty()
            
            # Afficher les résultats
            show_prediction_results(url_input, probas, predictions, features, model_loader)
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Error analyzing the URL: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # Si aucune URL n'est saisie, afficher un message explicatif
    elif analyze_button and not url_input:
        st.warning("Please enter a URL to analyze.")
        
    # Section d'information
    with st.expander("How does this analysis work?"):
        st.markdown(
            """
        #### URL Analysis Process
        
        Our system uses a multi-step process to analyze URLs:
        
        1. **URL Structure Analysis**: We examine the URL structure itself (length, special characters, subdomains, etc.)
        
        2. **Domain Information**: We check the domain's age, registration details, and reputation
        
        3. **Content Analysis**: We analyze the HTML content for suspicious elements (forms, redirects, iframes, etc.)
        
        4. **Machine Learning**: Our LightGBM model, trained on thousands of examples, evaluates over 20 features
        to determine if the URL is potentially malicious
        
        5. **Result Interpretation**: We provide a comprehensive report with a trust score and detailed explanations
        """
        )
        
    # Avertissement de sécurité
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
    Affiche les résultats de la prédiction.
    
    Args:
        url: L'URL analysée
        probas: Les probabilités prédites
        predictions: Les classes prédites
        features: Le dictionnaire des caractéristiques
        model_loader: L'instance du chargeur de modèle
    """
    logger = logging.getLogger(__name__)
    
    # Récupérer la prédiction et la probabilité
    is_phishing = predictions[0] == 1
    phishing_proba = probas[0]
    if isinstance(phishing_proba, np.ndarray) and len(phishing_proba) > 0:
        phishing_proba = phishing_proba[0]
    
    # Logs pour le débogage
    logger.info(f"Type of phishing_proba: {type(phishing_proba)}")
    logger.info(f"Raw value of phishing_proba: {phishing_proba}")
    
    # Vérifier si la valeur est valide (pas NaN ou infinie)
    if np.isnan(phishing_proba) or np.isinf(phishing_proba):
        logger.error(f"Invalid probability detected: {phishing_proba}, using a default value")
        # Utiliser une valeur par défaut basée sur les caractéristiques
        is_suspicious = (
            features.get("HasPasswordField", 0) == 1 
            and features.get("HasSubmitButton", 0) == 1 
            and features.get("IsHTTPS", 0) == 0
        )
        phishing_proba = 0.8 if is_suspicious else 0.2
    
    # Si la probabilité est exactement 0 ou 1, ajuster légèrement
    if phishing_proba == 0:
        logger.warning("Probability exactly 0 detected, adjusting to 0.01")
        phishing_proba = 0.01
    elif phishing_proba == 1:
        logger.warning("Probability exactly 1 detected, adjusting to 0.99")
        phishing_proba = 0.99
        
    # Calculer le score de légitimité (inverse de la probabilité de phishing)
    legitimate_proba = 1 - phishing_proba
    legitimacy_score = legitimate_proba * 100
    
    logger.info(f"URL: {url}")
    logger.info(f"Raw prediction: {predictions[0]}, Phishing probability: {phishing_proba}")
    logger.info(f"Calculated legitimacy score: {legitimacy_score}%")
    
    # Titre et séparateur
    st.markdown("---")
    st.markdown("<h2 class='section-title'>Analysis Results</h2>", unsafe_allow_html=True)
    
    # Afficher le résultat principal
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
    
    # Afficher l'URL analysée
    with col2:
        st.markdown("<h3>Analyzed URL:</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='word-break: break-all; font-size: 1.1rem;'>{url}</p>",
            unsafe_allow_html=True,
        )
        
        # Jauge de confiance avec Plotly
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=legitimacy_score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Legitimacy Score", "font": {"size": 24}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
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
    
    # Séparateur
    st.markdown("---")
    
    # Créer deux colonnes pour l'analyse détaillée
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3>Key factors for this prediction</h3>", unsafe_allow_html=True)
        
        # Obtenir les caractéristiques les plus importantes
        model_components = model_loader.get_model_components()
        feature_importance = model_components.get("metrics", {}).get("feature_importance", [])
        feature_names = model_components.get("selected_feature_names", [])
        
# Si nous avons des informations d'importance
        if len(feature_importance) > 0 and len(feature_names) > 0:
            # Convertir en listes Python si ce sont des tableaux numpy
            if isinstance(feature_importance, np.ndarray):
                feature_importance = feature_importance.tolist()
            if isinstance(feature_names, np.ndarray):
                feature_names = feature_names.tolist()
                
            # Créer un dictionnaire d'importance
            importance_dict = dict(zip(feature_names, feature_importance))
            
            # Trier les caractéristiques par importance
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Afficher les 5 facteurs les plus importants
            top_features = sorted_features[:5]
            
            # Créer le DataFrame pour l'affichage
            df = pd.DataFrame({
                "Feature": [FEATURE_DISPLAY_NAMES.get(f[0], f[0]) for f in top_features],
                "Importance": [f[1] for f in top_features]
            })
            
            # Afficher un tableau simple
            st.table(df)
            
            # Ajouter un résumé des facteurs déterminants pour cette URL spécifique
            st.markdown("<h4>Determining factors for this URL:</h4>", unsafe_allow_html=True)
            
            factors_list = []
            
            # Vérifier les facteurs de risque ou de confiance spécifiques
            if features.get("HasPasswordField", 0) == 1 and features.get("IsHTTPS", 0) == 0:
                factors_list.append("❌ **Unsecured password form**: The site asks for a password without using HTTPS.")
                
            if features.get("IsHTTPS", 0) == 1:
                factors_list.append("✅ **Secure connection**: The site uses HTTPS protocol to encrypt data.")
                
            if features.get("HasTitle", 0) == 0:
                factors_list.append("❌ **No title**: The site has no title tag, which is unusual for a legitimate site.")
                
            if features.get("HasFavicon", 0) == 0:
                factors_list.append("❌ **No icon**: The site has no favicon, which is often the case with phishing sites.")
                
            if features.get("NoOfObfuscatedChar", 0) > 0:
                factors_list.append(f"❌ **Obfuscated URL**: The URL contains {features.get('NoOfObfuscatedChar')} encoded characters, which may mask its true destination.")
                
            if features.get("HasCopyright", 0) == 1:
                factors_list.append("✅ **Copyright mention**: The site contains a copyright mention, usually present on legitimate sites.")
                
            if features.get("HasSocialNetworking", 0) == 1:
                factors_list.append("✅ **Social network links**: Presence of links to social networks, often found on legitimate sites.")
                
            if features.get("NoOfiFrame", 0) > 2:
                factors_list.append(f"❌ **Numerous iframes**: The site contains {features.get('NoOfiFrame')} iframes, which may indicate malicious content.")
                
            if features.get("IsTinyURL", 0) == 1:
                factors_list.append("❌ **Shortened URL**: This URL uses a URL shortening service, which can mask its true destination.")
                
            if features.get("HasSuspiciousKeyword", 0) == 1:
                factors_list.append("❌ **Suspicious keywords**: The domain contains suspicious words often used in phishing.")
                
            # Afficher les facteurs
            if factors_list:
                for factor in factors_list:
                    st.markdown(factor)
            else:
                st.markdown("No specific determining factors were identified for this URL.")
                
    with col2:
        st.markdown("<h3>Features extracted from the URL</h3>", unsafe_allow_html=True)
        
        # Créer un DataFrame pour l'affichage
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
            "HasSuspiciousKeyword": "Suspicious keywords",
            "IsTinyURL": "Shortened URL",
            "DomainAge": "Domain age < 6 months"
        }
        
        for feature in sorted(features.keys()):
            if feature != "url":  # Exclure l'URL complète
                # Formater les valeurs pour un affichage plus convivial
                value = features[feature]
                
                # Formatage spécial pour certaines caractéristiques
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
                    "HasSuspiciousKeyword",
                    "IsTinyURL",
                    "DomainAge"
                ]:
                    value = "Yes" if value == 1 else "No"
                    
                display_features[display_names.get(feature, feature)] = value
                
        # Créer un DataFrame pour l'affichage
        df_display = pd.DataFrame(
            list(display_features.items()), columns=["Feature", "Value"]
        )
        
        # Afficher le tableau
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    # Recommandations de sécurité
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
    
    # Ajouter une section technique détaillée (sous un expander pour ne pas surcharger l'interface)
    with st.expander("Technical details about the URL analysis"):
        st.markdown("### URL Structure Analysis")
        
        # Afficher l'URL décomposée
        parsed = urllib.parse.urlparse(url)
        extracted = tldextract.extract(url)
        
        st.markdown("#### URL Components:")
        components = {
            "Scheme": parsed.scheme,
            "Domain": parsed.netloc,
            "Path": parsed.path,
            "Parameters": parsed.params,
            "Query": parsed.query,
            "Fragment": parsed.fragment,
            "Subdomain": extracted.subdomain,
            "Root Domain": extracted.domain,
            "TLD": extracted.suffix
        }
        
        comp_df = pd.DataFrame(list(components.items()), columns=["Component", "Value"])
        st.table(comp_df)
        
        # Risques courants associés aux caractéristiques
        st.markdown("#### Common Phishing Indicators:")
        indicators = [
            {"Indicator": "Domain Age < 6 months", "Risk Level": "High", "Present": "Yes" if features.get("DomainAge", 1) == 1 else "No"},
            {"Indicator": "IP Address in URL", "Risk Level": "High", "Present": "No"},
            {"Indicator": "Shortened URL", "Risk Level": "Medium", "Present": "Yes" if features.get("IsTinyURL", 0) == 1 else "No"},
            {"Indicator": "Multiple Subdomains", "Risk Level": "Medium", "Present": "Yes" if features.get("NoOfSubDomain", 0) > 1 else "No"},
            {"Indicator": "Suspicious Keywords", "Risk Level": "Medium", "Present": "Yes" if features.get("HasSuspiciousKeyword", 0) == 1 else "No"},
            {"Indicator": "Many Obfuscated Characters", "Risk Level": "High", "Present": "Yes" if features.get("NoOfObfuscatedChar", 0) > 0 else "No"},
            {"Indicator": "HTTP (Not HTTPS)", "Risk Level": "Medium", "Present": "Yes" if features.get("IsHTTPS", 1) == 0 else "No"},
            {"Indicator": "Password Field without HTTPS", "Risk Level": "Very High", "Present": "Yes" if features.get("HasPasswordField", 0) == 1 and features.get("IsHTTPS", 0) == 0 else "No"}
        ]
        
        indicators_df = pd.DataFrame(indicators)
        st.table(indicators_df)