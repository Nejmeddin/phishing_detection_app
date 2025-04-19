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
import socket
import ssl
from bs4 import BeautifulSoup
from datetime import datetime
import time

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WebFeatureExtractor:
    """
    Class to extract relevant features from URLs and HTML content
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
        ]

        # Verify that all required features are in the expected list
        missing = set(self.required_features) - set(self.expected_features)
        if missing:
            logger.warning(
                f"Some required features are not in the expected list: {missing}"
            )

        # Keywords for detecting specific features
        self.bank_keywords = [
            "bank",
            "credit",
            "debit",
            "account",
            "card",
            "balance",
            "transfer",
        ]
        self.pay_keywords = ["pay", "payment", "transaction", "bill", "invoice"]
        self.crypto_keywords = [
            "crypto",
            "bitcoin",
            "ethereum",
            "wallet",
            "blockchain",
            "token",
        ]
        self.social_networks = [
            "facebook",
            "twitter",
            "instagram",
            "linkedin",
            "youtube",
            "pinterest",
        ]

        # Classify features as URL and HTML
        self.url_features = [
            "IsHTTPS",
            "IsDomainIP",
            "URLLength",
            "NoOfSubDomain",
            "NoOfDots",
            "NoOfObfuscatedChar",
            "NoOfEqual",
            "NoOfQmark",
            "NoOfAmp",
            "NoOfDigits",
        ]

        self.html_features = [
            "LineLength",
            "HasTitle",
            "HasMeta",
            "HasFavicon",
            "HasExternalFormSubmit",
            "HasCopyright",
            "HasSocialNetworking",
            "HasPasswordField",
            "HasSubmitButton",
            "HasKeywordBank",
            "HasKeywordPay",
            "HasKeywordCrypto",
            "NoOfPopup",
            "NoOfiFrame",
            "NoOfImage",
            "NoOfJS",
            "NoOfCSS",
            "NoOfURLRedirect",
            "NoOfHyperlink",
        ]

        # Create specific lists for required features
        self.required_url_features = [
            f for f in self.url_features if f in self.required_features
        ]
        self.required_html_features = [
            f for f in self.html_features if f in self.required_features
        ]

        logger.info(f"URL features to extract: {self.required_url_features}")
        logger.info(f"HTML features to extract: {self.required_html_features}")

        # Dictionary of default values for each feature
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

        # Setup a request session with defaults
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )
        # Set timeout and retry strategy
        self.timeout = 10  # seconds
        self.max_retries = 2

    def extract_features(self, url: str) -> Dict[str, Union[int, float, str]]:
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
            if self.required_html_features:
                try:
                    html_features = self._extract_html_features_with_retry(url)
                    features.update(html_features)
                except Exception as e:
                    logger.error(f"Error extracting HTML features: {str(e)}")
                    # Generate default values for HTML features
                    features.update(self._generate_default_html_features())

            # 3. Ensure all required features are present with appropriate defaults
            for feature in self.required_features:
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

        # Calculate required URL features
        if "IsHTTPS" in self.required_url_features:
            features["IsHTTPS"] = 1 if parsed_url.scheme == "https" else 0

        if "URLLength" in self.required_url_features:
            features["URLLength"] = len(url)

        if "IsDomainIP" in self.required_url_features:
            features["IsDomainIP"] = 1 if self._is_ip_address(parsed_url.netloc) else 0

        if "NoOfSubDomain" in self.required_url_features:
            subdomain_count = (
                len(extract_result.subdomain.split("."))
                if extract_result.subdomain
                else 0
            )
            features["NoOfSubDomain"] = subdomain_count

        if "NoOfDots" in self.required_url_features:
            features["NoOfDots"] = url.count(".")

        if "NoOfObfuscatedChar" in self.required_url_features:
            features["NoOfObfuscatedChar"] = len(re.findall(r"%[0-9a-fA-F]{2}", url))

        if "NoOfQmark" in self.required_url_features:
            features["NoOfQmark"] = url.count("?")

        if "NoOfDigits" in self.required_url_features:
            features["NoOfDigits"] = sum(c.isdigit() for c in url)

        return features

    def _extract_html_features_with_retry(self, url):
        """
        Extract features from the HTML content with retry mechanism.

        Args:
            url: The URL to analyze

        Returns:
            Dict: Features extracted from the HTML
        """
        # Initialize empty features dict
        features = {}

        # If no HTML features are required, return an empty dictionary
        if not self.required_html_features:
            return features

        # Implement retry mechanism
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, timeout=self.timeout)
                html_content = response.text
                soup = BeautifulSoup(html_content, "html.parser")

                # Extract all HTML features
                features = self._process_html_content(html_content, soup)

                # If we got here, we succeeded
                break

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt+1} failed: {str(e)}")
                if attempt < self.max_retries:
                    # Wait before retrying (exponential backoff)
                    wait_time = 2**attempt
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {self.max_retries+1} attempts failed")
                    raise

        return features

    def _process_html_content(self, html_content, soup):
        """
        Process HTML content to extract features.

        Args:
            html_content: The HTML content
            soup: The BeautifulSoup object

        Returns:
            Dict: Features extracted from the HTML
        """
        features = {}

        # Extract features as needed
        if "LineLength" in self.required_html_features:
            features["LineLength"] = len(html_content.splitlines())

        if "HasTitle" in self.required_html_features:
            features["HasTitle"] = 1 if soup.title else 0

        if "HasMeta" in self.required_html_features:
            features["HasMeta"] = 1 if soup.find_all("meta") else 0

        if "HasFavicon" in self.required_html_features:
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

        if "HasCopyright" in self.required_html_features:
            has_copyright = 0
            if "©" in html_content or "copyright" in html_content.lower():
                has_copyright = 1
            features["HasCopyright"] = has_copyright

        if "HasSocialNetworking" in self.required_html_features:
            has_social = 0
            for network in self.social_networks:
                if network in html_content.lower():
                    has_social = 1
                    break
            features["HasSocialNetworking"] = has_social

        if "HasPasswordField" in self.required_html_features:
            password_fields = soup.find_all("input", {"type": "password"})
            features["HasPasswordField"] = 1 if password_fields else 0

        if "HasSubmitButton" in self.required_html_features:
            submit_buttons = soup.find_all("input", {"type": "submit"})
            submit_buttons.extend(soup.find_all("button", {"type": "submit"}))
            features["HasSubmitButton"] = 1 if submit_buttons else 0

        if "HasKeywordCrypto" in self.required_html_features:
            has_crypto_keyword = 0
            for keyword in self.crypto_keywords:
                if keyword in html_content.lower():
                    has_crypto_keyword = 1
                    break
            features["HasKeywordCrypto"] = has_crypto_keyword

        if "NoOfPopup" in self.required_html_features:
            popups = 0
            popups += len(
                soup.find_all("script", string=re.compile("window.open|popup|alert"))
            )
            features["NoOfPopup"] = popups

        if "NoOfiFrame" in self.required_html_features:
            iframes = len(soup.find_all("iframe"))
            features["NoOfiFrame"] = iframes

        if "NoOfImage" in self.required_html_features:
            images = len(soup.find_all("img"))
            features["NoOfImage"] = images

        if "NoOfJS" in self.required_html_features:
            scripts = len(soup.find_all("script"))
            features["NoOfJS"] = scripts

        if "NoOfCSS" in self.required_html_features:
            css = len(soup.find_all("link", {"rel": "stylesheet"}))
            css += len(soup.find_all("style"))
            features["NoOfCSS"] = css

        if "NoOfURLRedirect" in self.required_html_features:
            redirects = 0
            redirect_patterns = ["window.location", "document.location", ".href"]
            scripts = soup.find_all("script")
            for script in scripts:
                if script.string:
                    for pattern in redirect_patterns:
                        if pattern in script.string:
                            redirects += 1
            features["NoOfURLRedirect"] = redirects

        if "NoOfHyperlink" in self.required_html_features:
            links = len(soup.find_all("a"))
            features["NoOfHyperlink"] = links

        return features

    def _is_ip_address(self, domain):
        """Checks if a domain is an IP address."""
        pattern = re.compile(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")
        if not pattern.match(domain):
            return False
        return all(0 <= int(n) < 256 for n in pattern.match(domain).groups())

    def _generate_default_features(self):
        """
        Generates a dictionary of default features based on the default_values dictionary.
        """
        features = {"url": "error"}

        # Add default values for all required features from our defaults dictionary
        for feature in self.required_features:
            features[feature] = self.default_values.get(feature, 0)

        return features

    def _generate_default_html_features(self):
        """
        Generates default values for HTML features.
        """
        features = {}

        for feature in self.required_html_features:
            features[feature] = self.default_values.get(feature, 0)

        return features

    def ensure_all_features(self, features_dict: Dict) -> Dict:
        """
        Ensures that all features required by the model are present.

        Args:
            features_dict: Dictionary of extracted features

        Returns:
            Dict: Dictionary completed with all required features
        """
        # Check for missing features
        missing_features = set(self.required_features) - set(features_dict.keys())

        if missing_features:
            logger.warning(f"Missing features in final result: {missing_features}")

            # Add missing features with appropriate default values
            for feature in missing_features:
                features_dict[feature] = self.default_values.get(feature, 0)

        return features_dict
