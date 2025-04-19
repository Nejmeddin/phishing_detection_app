"""
Module pour enrichir les caractéristiques d'URL manquantes via des APIs gratuites
"""

import re
import json
import logging
import requests
import socket
import ssl
import tldextract
import whois
import time
import urllib.parse
from datetime import datetime
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Union, Set

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureEnrichment:
    """
    Classe pour enrichir les caractéristiques manquantes d'une URL
    en utilisant des APIs et services gratuits
    """

    def __init__(self):
        """Initialise l'enrichisseur de caractéristiques"""
        self.api_cache = {}  # Cache simple pour éviter les appels répétés

        # Configuration des APIs
        self.virustotal_api_key = (
            "0a3830307e8a38cf1b21c2d80ab063764bdb8252f86c8a67fcfac37ad8c55e18"
        )
        self.whoisxml_api_key = "at_XsISAbfhEZlaeFi72Isz9nUKI6Qk0"

        # URLs de base pour les APIs
        self.virustotal_base_url = "https://www.virustotal.com/api/v3"
        self.urlscan_base_url = "https://urlscan.io/api/v1"
        self.whoisxml_base_url = "https://www.whoisxmlapi.com/whoisserver/WhoisService"

        # Dictionnaire des valeurs par défaut spécifiques à chaque feature
        self.default_values = {
            "IsHTTPS": 0,  # Default: not HTTPS
            "URLLength": 0,  # Default: zero length
            "NoOfSubDomain": 0,  # Default: no subdomains
            "NoOfDots": 0,  # Default: no dots
            "NoOfObfuscatedChar": 0,  # Default: no obfuscated chars
            "NoOfQmark": 0,  # Default: no question marks
            "NoOfDigits": 0,  # Default: no digits
            "LineLength": 0,  # Default: no lines
            "HasTitle": 0,  # Default: no title
            "HasMeta": 0,  # Default: no meta
            "HasFavicon": 0,  # Default: no favicon
            "HasCopyright": 0,  # Default: no copyright
            "HasSocialNetworking": 0,  # Default: no social networking
            "HasPasswordField": 0,  # Default: no password field
            "HasSubmitButton": 0,  # Default: no submit button
            "HasKeywordCrypto": 0,  # Default: no crypto keywords
            "NoOfPopup": 0,  # Default: no popups
            "NoOfiFrame": 0,  # Default: no iframes
            "NoOfImage": 0,  # Default: no images
            "NoOfJS": 0,  # Default: no JS
            "NoOfCSS": 0,  # Default: no CSS
            "NoOfURLRedirect": 0,  # Default: no redirects
            "NoOfHyperlink": 0,  # Default: no hyperlinks
        }

    def enrich_features(
        self, url: str, features: Dict[str, Any], missing_features: List[str]
    ) -> Dict[str, Any]:
        """
        Enrichit les caractéristiques manquantes en utilisant diverses sources

        Args:
            url: L'URL à analyser
            features: Dictionnaire des caractéristiques déjà extraites
            missing_features: Liste des caractéristiques manquantes à enrichir

        Returns:
            Dict: Caractéristiques enrichies
        """
        logger.info(
            f"Enrichissement des caractéristiques manquantes: {missing_features}"
        )

        # Initialiser un dictionnaire pour les caractéristiques enrichies
        enriched = {}

        # Si l'URL est mal formée, peu de choses peuvent être faites
        if not url or not isinstance(url, str):
            return self._generate_default_values(missing_features)

        # Normaliser l'URL
        if not url.startswith(("http://", "https://")):
            url = "http://" + url

        try:
            # 1. Try VT API for comprehensive data (if available)
            if (
                len(missing_features) > 3
            ):  # Si plusieurs caractéristiques manquent, essayer VT d'abord
                vt_enriched = self._try_virustotal_api(url, missing_features)
                if vt_enriched:
                    enriched.update(vt_enriched)
                    # Mettre à jour la liste des caractéristiques manquantes
                    missing_features = [
                        f for f in missing_features if f not in enriched
                    ]

            # 2. Try WhoisXML API for domain info
            if any(f in missing_features for f in ["IsHTTPS"]):
                whois_enriched = self._try_whoisxml_api(url, missing_features)
                if whois_enriched:
                    enriched.update(whois_enriched)
                    # Mettre à jour la liste des caractéristiques manquantes
                    missing_features = [
                        f for f in missing_features if f not in enriched
                    ]

            # 3. Essayer d'obtenir des caractéristiques HTML via une API alternative si nécessaire
            html_features = set(
                [
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
            )

            missing_html = [f for f in missing_features if f in html_features]
            if missing_html:
                urlscan_enriched = self._try_urlscan_for_html(url, missing_html)
                enriched.update(urlscan_enriched)

                # Retirer les caractéristiques qui ont été enrichies
                missing_features = [f for f in missing_features if f not in enriched]

            # 4. Essayer d'obtenir des informations sur le domaine si nécessaire
            domain_features = ["IsHTTPS"]
            missing_domain = [f for f in missing_features if f in domain_features]
            if missing_domain:
                domain_enriched = self._get_domain_features(url, missing_domain)
                enriched.update(domain_enriched)

                # Retirer les caractéristiques qui ont été enrichies
                missing_features = [f for f in missing_features if f not in enriched]

            # 5. Calculer des caractéristiques URL basiques si nécessaire
            url_features = [
                "URLLength",
                "NoOfSubDomain",
                "NoOfDots",
                "NoOfObfuscatedChar",
                "NoOfQmark",
                "NoOfDigits",
            ]
            missing_url = [f for f in missing_features if f in url_features]
            if missing_url:
                url_enriched = self._calculate_url_features(url, missing_url)
                enriched.update(url_enriched)

                # Retirer les caractéristiques qui ont été enrichies
                missing_features = [f for f in missing_features if f not in enriched]

            # 6. Ajouter des valeurs par défaut pour toutes les caractéristiques restantes
            if missing_features:
                logger.warning(
                    f"Utilisation de valeurs par défaut pour: {missing_features}"
                )
                for feature in missing_features:
                    enriched[feature] = self.default_values.get(feature, 0)

            logger.info(f"Caractéristiques enrichies: {list(enriched.keys())}")
            return enriched

        except Exception as e:
            logger.error(
                f"Erreur lors de l'enrichissement des caractéristiques: {str(e)}"
            )
            import traceback

            logger.error(traceback.format_exc())
            return self._generate_default_values(missing_features)

    def _try_virustotal_api(
        self, url: str, missing_features: List[str]
    ) -> Dict[str, Any]:
        """
        Essaie d'obtenir des informations via l'API VirusTotal.

        Args:
            url: L'URL à analyser
            missing_features: Liste des caractéristiques manquantes

        Returns:
            Dict: Caractéristiques enrichies par VirusTotal
        """
        enriched = {}

        # Vérifier le cache
        cache_key = f"virustotal_{url}"
        if cache_key in self.api_cache:
            data = self.api_cache[cache_key]
        else:
            try:
                # Encoder l'URL pour l'utiliser dans l'URL de l'API
                encoded_url = urllib.parse.quote_plus(url)
                api_url = f"{self.virustotal_base_url}/urls/{encoded_url}"

                # En-têtes pour l'API
                headers = {
                    "x-apikey": self.virustotal_api_key,
                    "Accept": "application/json",
                }

                # Faire la requête à l'API
                response = requests.get(api_url, headers=headers, timeout=10)

                # Vérifier si l'URL n'est pas trouvée (peut nécessiter une soumission)
                if response.status_code == 404:
                    # Soumettre l'URL pour analyse
                    submit_url = f"{self.virustotal_base_url}/urls"
                    payload = {"url": url}
                    submit_response = requests.post(
                        submit_url, headers=headers, data=payload, timeout=10
                    )

                    if submit_response.status_code == 200:
                        logger.info(f"URL soumise à VirusTotal pour analyse: {url}")
                        # Attendre quelques secondes pour permettre l'analyse
                        time.sleep(5)
                        # Réessayer de récupérer les résultats
                        response = requests.get(api_url, headers=headers, timeout=10)

                if response.status_code != 200:
                    logger.warning(
                        f"Erreur lors de la requête à VirusTotal: {response.status_code}"
                    )
                    return enriched

                data = response.json()
                self.api_cache[cache_key] = data

            except Exception as e:
                logger.error(f"Erreur lors de l'appel à VirusTotal: {str(e)}")
                return enriched

        try:
            # Extraire les attributs pertinents des données
            attributes = data.get("data", {}).get("attributes", {})

            # Obtenir les informations de last_analysis_stats
            stats = attributes.get("last_analysis_stats", {})

            # Extraire les caractéristiques HTML si disponibles
            html_content = attributes.get("last_http_response_content", "")

            if html_content:
                soup = BeautifulSoup(html_content, "html.parser")

                # Extraire les features HTML
                if "HasTitle" in missing_features:
                    enriched["HasTitle"] = 1 if soup.title else 0

                if "HasMeta" in missing_features:
                    enriched["HasMeta"] = 1 if soup.find_all("meta") else 0

                if "HasFavicon" in missing_features:
                    links = soup.find_all("link")
                    has_favicon = 0
                    for link in links:
                        rel = link.get("rel", "")
                        if isinstance(rel, list):
                            rel = " ".join(rel)
                        if "icon" in rel.lower():
                            has_favicon = 1
                            break
                    enriched["HasFavicon"] = has_favicon

                if "LineLength" in missing_features:
                    enriched["LineLength"] = len(html_content.splitlines())

                if "HasCopyright" in missing_features:
                    enriched["HasCopyright"] = (
                        1
                        if "©" in html_content or "copyright" in html_content.lower()
                        else 0
                    )

                if "HasSocialNetworking" in missing_features:
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
                    enriched["HasSocialNetworking"] = has_social

                if "HasPasswordField" in missing_features:
                    password_fields = soup.find_all("input", {"type": "password"})
                    enriched["HasPasswordField"] = 1 if password_fields else 0

                if "HasSubmitButton" in missing_features:
                    submit_buttons = soup.find_all("input", {"type": "submit"})
                    submit_buttons.extend(soup.find_all("button", {"type": "submit"}))
                    enriched["HasSubmitButton"] = 1 if submit_buttons else 0

                if "HasKeywordCrypto" in missing_features:
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
                    enriched["HasKeywordCrypto"] = has_crypto

                if "NoOfPopup" in missing_features:
                    popups = len(
                        soup.find_all(
                            "script", string=re.compile("window.open|popup|alert")
                        )
                    )
                    enriched["NoOfPopup"] = popups

                if "NoOfiFrame" in missing_features:
                    enriched["NoOfiFrame"] = len(soup.find_all("iframe"))

                if "NoOfImage" in missing_features:
                    enriched["NoOfImage"] = len(soup.find_all("img"))

                if "NoOfJS" in missing_features:
                    enriched["NoOfJS"] = len(soup.find_all("script"))

                if "NoOfCSS" in missing_features:
                    css_count = len(soup.find_all("link", {"rel": "stylesheet"}))
                    css_count += len(soup.find_all("style"))
                    enriched["NoOfCSS"] = css_count

                if "NoOfURLRedirect" in missing_features:
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
                    enriched["NoOfURLRedirect"] = redirects

                if "NoOfHyperlink" in missing_features:
                    enriched["NoOfHyperlink"] = len(soup.find_all("a"))

            # HTTPS information
            if "IsHTTPS" in missing_features:
                last_analysis_url = attributes.get("last_analysis_url", "")
                enriched["IsHTTPS"] = (
                    1 if last_analysis_url.startswith("https://") else 0
                )

            return enriched

        except Exception as e:
            logger.error(f"Erreur lors du traitement des données VirusTotal: {str(e)}")
            return enriched
