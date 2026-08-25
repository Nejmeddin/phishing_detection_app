"""Extract model features from a URL and the page it points to.

Feature extraction runs in two halves. The lexical half works on the URL string
alone and always succeeds. The content half needs to fetch the page, which is
exactly what a phishing host is most likely to prevent, so every failure there
degrades to a neutral default instead of aborting the analysis.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.parse
from datetime import datetime
from typing import Any

import requests
import tldextract
import whois

from src.config import (
    DEFAULT_FEATURE_VALUES,
    EXPECTED_FEATURES,
    HTML_FEATURES,
    REQUEST_USER_AGENT,
    URL_FEATURES_CONFIG,
)
from src.preprocessing.html_features import extract_html_features

logger = logging.getLogger(__name__)

# Domains whose whole purpose is to hide the destination behind a short alias.
SHORTENING_SERVICES = re.compile(
    "|".join(re.escape(domain) for domain in URL_FEATURES_CONFIG["shorteners"]),
    re.IGNORECASE,
)

# Trust-baiting vocabulary that phishing domains lean on heavily.
SENSITIVE_WORDS = URL_FEATURES_CONFIG["suspicious_keywords"]

# A domain younger than this is treated as risky by the model.
NEW_DOMAIN_THRESHOLD_DAYS = 180


class EnhancedFeatureExtractor:
    """Turn a URL into the feature vector the LightGBM model expects."""

    def __init__(self, required_features: list[str] | None = None):
        """
        Args:
            required_features: Features the loaded model asks for. Used for
                logging and diagnostics; extraction always produces the full
                expected set regardless.
        """
        self.required_features = required_features or []
        self.expected_features = list(EXPECTED_FEATURES) + [
            "IsTinyURL",
            "HasSuspiciousKeyword",
        ]
        self.default_values = dict(DEFAULT_FEATURE_VALUES)
        self.default_values.update({"IsTinyURL": 0, "HasSuspiciousKeyword": 0})

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": REQUEST_USER_AGENT})
        self.timeout = 5  # seconds
        self.max_retries = 1

        logger.info("Features required by the model: %s", self.required_features)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def extract_features(self, url: str) -> dict[str, Any]:
        """Extract every expected feature for ``url``.

        Args:
            url: The URL to analyse, with or without a scheme.

        Returns:
            A feature mapping that always contains each expected feature, plus
            the original ``url``.
        """
        logger.info("Extracting features for URL: %s", url)
        features: dict[str, Any] = {"url": url}

        normalised = url if url.startswith(("http://", "https://")) else f"http://{url}"

        try:
            parsed_url = urllib.parse.urlparse(normalised)
            extract_result = tldextract.extract(normalised)
        except ValueError as exc:
            logger.error("Could not parse URL %s: %s", url, exc)
            return self._generate_default_features(url)

        features.update(
            self._extract_url_features(normalised, parsed_url, extract_result)
        )
        features.update(self._fetch_html_features(normalised))

        for feature in self.expected_features:
            if feature not in features:
                logger.warning("Feature %s missing, using its default", feature)
                features[feature] = self.default_values.get(feature, 0)

        logger.info("Extraction complete: %d features", len(features) - 1)
        return features

    # ------------------------------------------------------------------ #
    # Lexical features
    # ------------------------------------------------------------------ #
    def _extract_url_features(
        self, url: str, parsed_url, extract_result
    ) -> dict[str, Any]:
        """Compute the features derivable from the URL string itself."""
        domain = parsed_url.netloc.lower()

        features: dict[str, Any] = {
            "IsHTTPS": 1 if parsed_url.scheme == "https" else 0,
            "URLLength": len(url),
            "NoOfSubDomain": (
                len(extract_result.subdomain.split("."))
                if extract_result.subdomain
                else 0
            ),
            "NoOfDots": url.count("."),
            "NoOfObfuscatedChar": len(re.findall(r"%[0-9a-fA-F]{2}", url)),
            "NoOfQmark": url.count("?"),
            "NoOfDigits": sum(character.isdigit() for character in url),
            "IsTinyURL": 1 if SHORTENING_SERVICES.search(url) else 0,
            "HasSuspiciousKeyword": (
                1 if any(word in domain for word in SENSITIVE_WORDS) else 0
            ),
        }

        domain_age = self._get_domain_age_flag(parsed_url.netloc)
        if domain_age is not None:
            features["DomainAge"] = domain_age

        return features

    def _get_domain_age_flag(self, domain: str) -> int | None:
        """Return 1 for a recently registered domain, 0 for an established one.

        Args:
            domain: The network location to look up.

        Returns:
            The risk flag, or ``None`` when WHOIS data is unavailable.
        """
        if not domain:
            return None

        try:
            record = whois.whois(domain)
        except Exception as exc:  # python-whois raises many undeclared types.
            logger.info("WHOIS lookup failed for %s: %s", domain, exc)
            return None

        creation_date = self._first_date(getattr(record, "creation_date", None))
        if creation_date is None:
            return None

        age_days = (datetime.now() - creation_date).days
        return 0 if age_days >= NEW_DOMAIN_THRESHOLD_DAYS else 1

    @staticmethod
    def _first_date(value: Any) -> datetime | None:
        """Normalise a WHOIS date field, which may be a list or absent."""
        if isinstance(value, list):
            value = value[0] if value else None
        return value if isinstance(value, datetime) else None

    # ------------------------------------------------------------------ #
    # Content features
    # ------------------------------------------------------------------ #
    def _fetch_html_features(self, url: str) -> dict[str, Any]:
        """Fetch the page and derive its content features.

        Args:
            url: The normalised URL to request.

        Returns:
            The HTML feature mapping, filled with defaults if the page could not
            be retrieved.
        """
        defaults = {name: self.default_values.get(name, 0) for name in HTML_FEATURES}

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, timeout=self.timeout)
                features = dict(defaults)
                features.update(extract_html_features(response.text))
                return features

            except requests.RequestException as exc:
                logger.warning("Request attempt %d failed: %s", attempt + 1, exc)
                if attempt < self.max_retries:
                    backoff = 2**attempt
                    logger.info("Retrying in %d seconds", backoff)
                    time.sleep(backoff)

        logger.error("Could not fetch %s, using default content features", url)
        return defaults

    # ------------------------------------------------------------------ #
    # Fallback
    # ------------------------------------------------------------------ #
    def _generate_default_features(self, url: str = "error") -> dict[str, Any]:
        """Return a complete, neutral feature vector."""
        features: dict[str, Any] = {"url": url}
        for feature in self.expected_features:
            features[feature] = self.default_values.get(feature, 0)
        return features
