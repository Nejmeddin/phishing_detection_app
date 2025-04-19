"""
Configuration centralisée pour l'application de détection de phishing.
Ce module contient tous les paramètres de configuration utilisés dans l'application.
"""

import os
from pathlib import Path

# Chemins de base
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Fichiers de données
PHISHING_DATASET_PATH = os.path.join(RAW_DATA_DIR, "Phishing_Legitimate_full.csv")
MODEL_PATH = os.path.join(PROCESSED_DATA_DIR, "lightgbm_phishing_model.pkl")
MODEL_METRICS_PATH = os.path.join(PROCESSED_DATA_DIR, "model_metrics.json")

# Configuration des APIs pour l'analyse d'URLs
VIRUSTOTAL_API_KEY = os.environ.get(
    "VIRUSTOTAL_API_KEY",
    "0a3830307e8a38cf1b21c2d80ab063764bdb8252f86c8a67fcfac37ad8c55e18",
)
URLSCAN_API_KEY = os.environ.get("URLSCAN_API_KEY", "")
WHOISXML_API_KEY = os.environ.get(
    "WHOISXML_API_KEY",
    "at_XsISAbfhEZlaeFi72Isz9nUKI6Qk0",
)

# URLs de base pour les APIs
VIRUSTOTAL_BASE_URL = "https://www.virustotal.com/api/v3"
URLSCAN_BASE_URL = "https://urlscan.io/api/v1"
WHOISXML_BASE_URL = "https://www.whoisxmlapi.com/whoisserver/WhoisService"

# Paramètres du modèle
MODEL_VERSION = "1.0.0"
PREDICTION_THRESHOLD = 0.5  # Seuil pour la classification binaire

# Configuration de l'extraction de caractéristiques
URL_FEATURES_CONFIG = {
    "suspicious_keywords": [
        "secure",
        "account",
        "webscr",
        "login",
        "ebayisapi",
        "signin",
        "banking",
        "confirm",
        "secure",
        "support",
        "update",
        "authenticate",
        "verification",
        "verify",
        "customer",
        "paypal",
        "password",
    ],
    "suspicious_tlds": [
        "xyz",
        "top",
        "club",
        "pw",
        "online",
        "site",
        "live",
        "win",
        "party",
        "stream",
        "racing",
        "loan",
        "jetzt",
        "work",
    ],
    "shorteners": [
        "bit.ly",
        "goo.gl",
        "t.co",
        "tinyurl.com",
        "is.gd",
        "cli.gs",
        "ow.ly",
        "tr.im",
        "u.to",
        "twitthis.com",
        "snipurl.com",
        "short.to",
        "lnkd.in",
    ],
}

# Configuration de l'interface utilisateur Streamlit
STREAMLIT_CONFIG = {
    "page_title": "Détection de Phishing",
    "page_icon": "🔒",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "theme_primary_color": "#1E90FF",
    "theme_secondary_color": "#3D85C6",
    "info_box_color": "#E8F4F8",
    "warning_box_color": "#FFEBEE",
    "success_box_color": "#E8F5E9",
}

# Noms d'affichage pour les caractéristiques
FEATURE_DISPLAY_NAMES = {
    "url_length": "Longueur de l'URL",
    "domain_length": "Longueur du domaine",
    "path_length": "Longueur du chemin",
    "query_length": "Longueur de la requête",
    "dots_count": "Nombre de points",
    "hyphens_count": "Nombre de tirets",
    "underscores_count": "Nombre de tirets bas",
    "slashes_count": "Nombre de barres obliques",
    "is_https": "Utilise HTTPS",
    "has_ip_address": "Contient une IP",
    "has_suspicious_tld": "Extension suspecte",
    "subdomain_count": "Nombre de sous-domaines",
    "domain_contains_number": "Domaine avec chiffres",
    "has_suspicious_keywords": "Mots-clés suspects",
    "is_tiny_url": "URL raccourcie",
    "domain_age": "Âge du domaine",
    "ssl_valid": "Certificat SSL valide",
    "is_blacklisted": "Site en liste noire",
    "has_port": "Utilise un port spécifique",
    "url_entropy": "Entropie de l'URL",
    "domain_entropy": "Entropie du domaine",
}

# Descriptions des caractéristiques pour le module d'explication
FEATURE_EXPLANATIONS = {
    "url_length": "Longueur totale de l'URL (les URLs de phishing sont souvent plus longues)",
    "domain_length": "Longueur du nom de domaine",
    "path_length": "Longueur du chemin dans l'URL (après le domaine)",
    "query_length": "Longueur de la partie requête de l'URL (après le '?')",
    "dots_count": "Nombre de points dans l'URL (les URLs de phishing ont souvent plus de sous-domaines)",
    "has_ip_address": "Utilisation d'une adresse IP au lieu d'un nom de domaine (signe suspect)",
    "has_suspicious_tld": "Utilisation d'une extension de domaine inhabituelle ou suspecte",
    "subdomain_count": "Nombre de sous-domaines (les attaquants utilisent souvent des sous-domaines multiples)",
    "domain_contains_number": "Présence de chiffres dans le nom de domaine (souvent utilisé pour l'usurpation d'identité)",
    "has_suspicious_keywords": "Présence de mots-clés suspects (login, compte, sécurité, etc.)",
    "domain_age": "Âge du domaine en années (les domaines récents sont plus suspects)",
    "is_https": "Utilisation du protocole HTTPS (sécurisé) ou HTTP (non sécurisé)",
    "ssl_valid": "Validité du certificat SSL",
    "alexa_rank": "Classement Alexa (popularité du site)",
    "is_blacklisted": "Présence dans des listes noires de sécurité",
    "is_tiny_url": "URL raccourcie via un service comme bit.ly (technique souvent utilisée pour masquer les URLs malveillantes)",
    "url_entropy": "Mesure de l'aléatoire dans l'URL (les URLs générées automatiquement ont une entropie élevée)",
    "domain_entropy": "Mesure de l'aléatoire dans le nom de domaine",
}
