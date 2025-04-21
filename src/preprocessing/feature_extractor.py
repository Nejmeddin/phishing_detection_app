"""
Module to extract features from a URL and its HTML content
for phishing detection.
"""

import re
import logging
import urllib.parse
import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Union, Optional, Set, Any
import tldextract
import whois
from datetime import datetime
import time
from bs4 import BeautifulSoup

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Liste des mots sensibles qui peuvent indiquer du phishing
SENSITIVE_WORDS = [
    "account",
    "confirm",
    "banking",
    "secure",
    "ebyisapi",
    "webscr",
    "signin",
    "mail",
    "install",
    "toolbar",
    "backup",
    "paypal",
    "password",
    "username",
    "verify",
    "update",
    "login",
    "support",
    "billing",
    "transaction",
    "security",
    "payment",
    "verify",
    "online",
    "customer",
    "service",
    "accountupdate",
    "verification",
    "important",
    "confidential",
    "limited",
    "access",
    "securitycheck",
    "verifyaccount",
    "information",
    "change",
    "notice",
    "myaccount",
    "updateinfo",
    "loginsecure",
    "protect",
    "transaction",
    "identity",
    "member",
    "personal",
    "actionrequired",
    "loginverify",
    "validate",
    "paymentupdate",
    "urgent",
]

# Services de raccourcissement d'URLs
SHORTENING_SERVICES = (
    r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|"
    r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|"
    r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"
    r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|"
    r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|"
    r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|"
    r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|"
    r"tr\.im|link\.zip\.net"
)


class EnhancedFeatureExtractor:
    """
    Class to extract enhanced features from URLs and HTML content
    for phishing detection.
    """

    def __init__(self, required_features=None):
        """
        Initialize the feature extractor.

        Args:
            required_features: List of features required by the model
        """
        # Store features required by the model
        self.required_features = required_features or []
        logger.info(f"Features required by the model: {self.required_features}")

        # Define all required features for the LightGBM model
        self.expected_features = [
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
            "IsTinyURL",
            "HasSuspiciousKeyword",
        ]

        # Setup a request session with defaults
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )
        # Set timeout and retry strategy
        self.timeout = 5  # seconds
        self.max_retries = 1

        # Dictionary of default values for each feature
        self.default_values = {
            "IsHTTPS": 0,
            "URLLength": 0,
            "NoOfSubDomain": 0,
            "NoOfDots": 0,
            "NoOfObfuscatedChar": 0,
            "NoOfQmark": 0,
            "NoOfDigits": 0,
            "LineLength": 0,
            "HasTitle": 0,
            "HasMeta": 0,
            "HasFavicon": 0,
            "HasCopyright": 0,
            "HasSocialNetworking": 0,
            "HasPasswordField": 0,
            "HasSubmitButton": 0,
            "HasKeywordCrypto": 0,
            "NoOfPopup": 0,
            "NoOfiFrame": 0,
            "NoOfImage": 0,
            "NoOfJS": 0,
            "NoOfCSS": 0,
            "NoOfURLRedirect": 0,
            "NoOfHyperlink": 0,
            "IsTinyURL": 0,
            "HasSuspiciousKeyword": 0,
        }

    def extract_features(self, url: str) -> Dict[str, Any]:
        """
        Extract all features from a URL and its HTML content.

        Args:
            url: The URL to analyze

        Returns:
            Dict: A dictionary containing the extracted features
        """
        try:
            logger.info(f"Extracting features for URL: {url}")
            features = {"url": url}

            # URL normalization
            if not url.startswith(("http://", "https://")):
                url = "http://" + url

            # Extract base components of the URL
            try:
                parsed_url = urllib.parse.urlparse(url)
                extract_result = tldextract.extract(url)
            except Exception as e:
                logger.error(f"Error parsing URL: {str(e)}")
                return self._generate_default_features()

            # 1. Extract URL features
            features.update(self._extract_url_features(url, parsed_url, extract_result))

            # 2. Extract HTML features with retries if needed
            html_features = self._extract_html_features_with_retry(url)
            features.update(html_features)

            # 3. Ensure all required features are present with appropriate defaults
            for feature in self.expected_features:
                if feature not in features:
                    logger.warning(f"Feature {feature} missing, using default value")
                    features[feature] = self.default_values.get(feature, 0)

            logger.info(
                f"Extraction successful: {len(features) - 1} features extracted"
            )
            return features

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            return self._generate_default_features()

    def _extract_url_features(self, url, parsed_url, extract_result):
        """
        Extract features from the URL.

        Args:
            url: The complete URL
            parsed_url: The parsed URL
            extract_result: The result from tldextract

        Returns:
            Dict: Features extracted from the URL
        """
        features = {}

        # IsHTTPS
        features["IsHTTPS"] = 1 if parsed_url.scheme == "https" else 0

        # URLLength
        features["URLLength"] = len(url)

        # Sous-domaines
        subdomain_count = (
            len(extract_result.subdomain.split(".")) if extract_result.subdomain else 0
        )
        features["NoOfSubDomain"] = subdomain_count

        # Compter les points
        features["NoOfDots"] = url.count(".")

        # Caractères obfusqués
        features["NoOfObfuscatedChar"] = len(re.findall(r"%[0-9a-fA-F]{2}", url))

        # Points d'interrogation
        features["NoOfQmark"] = url.count("?")

        # Chiffres dans l'URL
        features["NoOfDigits"] = sum(c.isdigit() for c in url)

        # URL raccourcie
        features["IsTinyURL"] = 1 if re.search(SHORTENING_SERVICES, url) else 0

        # Mots sensibles dans le domaine
        domain = parsed_url.netloc
        has_sensitive_word = 0
        for word in SENSITIVE_WORDS:
            if word in domain.lower():
                has_sensitive_word = 1
                break
        features["HasSuspiciousKeyword"] = has_sensitive_word

        # Tentative d'obtenir des informations sur le domaine
        try:
            domain_info = whois.whois(parsed_url.netloc)

            # Âge du domaine
            domain_age = 1  # Par défaut, considéré comme récent (suspect)
            if (
                domain_info
                and domain_info.creation_date
                and domain_info.expiration_date
            ):
                try:
                    # Gestion des cas où la date est une liste
                    creation_date = domain_info.creation_date
                    if isinstance(creation_date, list):
                        creation_date = creation_date[0]

                    expiration_date = domain_info.expiration_date
                    if isinstance(expiration_date, list):
                        expiration_date = expiration_date[0]

                    # Calculer l'âge en jours
                    if isinstance(creation_date, datetime) and isinstance(
                        expiration_date, datetime
                    ):
                        age_days = abs((expiration_date - creation_date).days)
                        # Considérer comme légitime si le domaine a plus de 6 mois
                        domain_age = 0 if (age_days / 30) >= 6 else 1
                except Exception as e:
                    logger.warning(
                        f"Erreur lors du calcul de l'âge du domaine: {str(e)}"
                    )

            features["DomainAge"] = domain_age

        except Exception as e:
            logger.warning(f"Impossible d'obtenir les informations whois: {str(e)}")

        return features

    def _extract_html_features_with_retry(self, url):
        """
        Extract features from the HTML content with retry mechanism.

        Args:
            url: The URL to analyze

        Returns:
            Dict: Features extracted from the HTML
        """
        # Caractéristiques HTML par défaut (en cas d'échec)
        default_html_features = {
            "LineLength": 0,
            "HasTitle": 0,
            "HasMeta": 0,
            "HasFavicon": 0,
            "HasCopyright": 0,
            "HasSocialNetworking": 0,
            "HasPasswordField": 0,
            "HasSubmitButton": 0,
            "HasKeywordCrypto": 0,
            "NoOfPopup": 0,
            "NoOfiFrame": 0,
            "NoOfImage": 0,
            "NoOfJS": 0,
            "NoOfCSS": 0,
            "NoOfURLRedirect": 0,
            "NoOfHyperlink": 0,
        }

        features = {}

        try:
            # Implement retry mechanism
            for attempt in range(self.max_retries + 1):
                try:
                    response = self.session.get(url, timeout=self.timeout)
                    html_content = response.text
                    soup = BeautifulSoup(html_content, "html.parser")

                    # Longueur du contenu
                    features["LineLength"] = len(html_content.splitlines())

                    # Présence d'un titre
                    features["HasTitle"] = 1 if soup.title else 0

                    # Présence de balises meta
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
                        1
                        if "©" in html_content or "copyright" in html_content.lower()
                        else 0
                    )

                    # Réseaux sociaux
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

                    # Champ de mot de passe
                    features["HasPasswordField"] = (
                        1 if soup.find_all("input", {"type": "password"}) else 0
                    )

                    # Bouton de soumission
                    submit_buttons = soup.find_all("input", {"type": "submit"})
                    submit_buttons.extend(soup.find_all("button", {"type": "submit"}))
                    features["HasSubmitButton"] = 1 if submit_buttons else 0

                    # Mots-clés crypto
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
                    popups = len(
                        soup.find_all(
                            "script", string=re.compile("window.open|popup|alert")
                        )
                    )
                    features["NoOfPopup"] = popups

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

                    # Redirections URL
                    redirects = 0
                    redirect_patterns = [
                        "window.location",
                        "document.location",
                        ".href",
                    ]
                    scripts = soup.find_all("script")
                    for script in scripts:
                        if script.string:
                            for pattern in redirect_patterns:
                                if pattern in script.string:
                                    redirects += 1
                    features["NoOfURLRedirect"] = redirects

                    # Hyperliens
                    features["NoOfHyperlink"] = len(soup.find_all("a"))

                    # Si on arrive ici, l'extraction a réussi
                    break

                except Exception as e:
                    logger.warning(f"Request attempt {attempt+1} failed: {str(e)}")
                    if attempt < self.max_retries:
                        # Wait before retrying (exponential backoff)
                        wait_time = 2**attempt
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {self.max_retries+1} attempts failed")
                        features.update(default_html_features)

        except Exception as e:
            logger.error(f"Error extracting HTML features: {str(e)}")
            features.update(default_html_features)

        # Vérifier que toutes les caractéristiques HTML sont présentes
        for feature, default_value in default_html_features.items():
            if feature not in features:
                features[feature] = default_value

        return features

    def _generate_default_features(self):
        """
        Generates a dictionary of default features based on the default_values dictionary.
        """
        features = {"url": "error"}

        # Add default values for all expected features from our defaults dictionary
        for feature in self.expected_features:
            features[feature] = self.default_values.get(feature, 0)

        return features
