"""Recover feature values that local extraction could not produce.

When :class:`~src.preprocessing.feature_extractor.EnhancedFeatureExtractor`
cannot reach a page — the host is down, blocks scrapers, or times out — the
model is still owed a complete feature vector. This module fills the gaps by
escalating through progressively cheaper sources:

1. VirusTotal, which may already hold a cached HTTP response for the URL;
2. WhoisXML, for domain-level facts;
3. urlscan.io, for a rendered snapshot of the page;
4. a direct HTTPS handshake, for transport-level facts;
5. pure string analysis of the URL, which never fails;
6. neutral defaults for whatever is still missing.

Every external call is optional: absent an API key the corresponding step is
skipped rather than failing, so the application degrades gracefully offline.
"""

from __future__ import annotations

import logging
import re
import socket
import ssl
import time
import urllib.parse
from typing import Any

import requests
import tldextract

from src.config import (
    DEFAULT_FEATURE_VALUES,
    HTML_FEATURES,
    REQUEST_TIMEOUT,
    REQUEST_USER_AGENT,
    URL_FEATURES,
    URLSCAN_API_KEY,
    URLSCAN_BASE_URL,
    VIRUSTOTAL_API_KEY,
    VIRUSTOTAL_BASE_URL,
    WHOISXML_API_KEY,
    WHOISXML_BASE_URL,
)
from src.preprocessing.html_features import extract_html_features

logger = logging.getLogger(__name__)


class FeatureEnrichment:
    """Fill in missing URL features using external services and heuristics."""

    def __init__(self) -> None:
        self.api_cache: dict[str, Any] = {}
        self.default_values = dict(DEFAULT_FEATURE_VALUES)

        self.virustotal_api_key = VIRUSTOTAL_API_KEY
        self.urlscan_api_key = URLSCAN_API_KEY
        self.whoisxml_api_key = WHOISXML_API_KEY

        self.virustotal_base_url = VIRUSTOTAL_BASE_URL
        self.urlscan_base_url = URLSCAN_BASE_URL
        self.whoisxml_base_url = WHOISXML_BASE_URL

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": REQUEST_USER_AGENT})

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def enrich_features(
        self, url: str, features: dict[str, Any], missing_features: list[str]
    ) -> dict[str, Any]:
        """Resolve ``missing_features`` for ``url``.

        Args:
            url: The URL under analysis.
            features: Features already extracted locally, used as context.
            missing_features: Names of the features still to be resolved.

        Returns:
            A mapping of every requested feature to a usable value. The keys of
            the result always cover ``missing_features``.
        """
        if not url or not isinstance(url, str):
            return self._generate_default_values(missing_features)

        remaining = list(missing_features)
        if not remaining:
            return {}

        logger.info("Enriching missing features: %s", remaining)

        if not url.startswith(("http://", "https://")):
            url = "http://" + url

        enriched: dict[str, Any] = {}

        try:
            # 1. VirusTotal often has a cached HTTP response covering many
            #    features at once, so it is worth one call when several are
            #    missing.
            if len(remaining) > 3 and self.virustotal_api_key:
                enriched.update(self._try_virustotal_api(url, remaining))
                remaining = [f for f in remaining if f not in enriched]

            # 2. Domain-level facts from WhoisXML.
            if remaining and self.whoisxml_api_key:
                enriched.update(self._try_whoisxml_api(url, remaining))
                remaining = [f for f in remaining if f not in enriched]

            # 3. A rendered snapshot from urlscan.io for HTML-derived features.
            missing_html = [f for f in remaining if f in HTML_FEATURES]
            if missing_html:
                enriched.update(self._try_urlscan_for_html(url, missing_html))
                remaining = [f for f in remaining if f not in enriched]

            # 4. Transport-level facts we can establish ourselves.
            missing_domain = [f for f in remaining if f in {"IsHTTPS"}]
            if missing_domain:
                enriched.update(self._get_domain_features(url, missing_domain))
                remaining = [f for f in remaining if f not in enriched]

            # 5. Pure string analysis — always available, never fails.
            missing_url = [f for f in remaining if f in URL_FEATURES]
            if missing_url:
                enriched.update(self._calculate_url_features(url, missing_url))
                remaining = [f for f in remaining if f not in enriched]

            # 6. Neutral defaults for anything left.
            if remaining:
                logger.warning("Falling back to default values for: %s", remaining)
                enriched.update(self._generate_default_values(remaining))

            logger.info("Enriched features: %s", sorted(enriched))
            return enriched

        except Exception:
            logger.exception("Feature enrichment failed for %s", url)
            # Never let enrichment break a prediction: return a complete,
            # neutral vector merged with whatever we did manage to resolve.
            fallback = self._generate_default_values(missing_features)
            fallback.update(enriched)
            return fallback

    # ------------------------------------------------------------------ #
    # Source 1: VirusTotal
    # ------------------------------------------------------------------ #
    def _try_virustotal_api(
        self, url: str, missing_features: list[str]
    ) -> dict[str, Any]:
        """Read a cached VirusTotal record for ``url``, if one exists."""
        enriched: dict[str, Any] = {}
        if not self.virustotal_api_key:
            return enriched

        cache_key = f"virustotal_{url}"
        if cache_key in self.api_cache:
            data = self.api_cache[cache_key]
        else:
            try:
                encoded_url = urllib.parse.quote_plus(url)
                api_url = f"{self.virustotal_base_url}/urls/{encoded_url}"
                headers = {
                    "x-apikey": self.virustotal_api_key,
                    "Accept": "application/json",
                }

                response = self.session.get(
                    api_url, headers=headers, timeout=REQUEST_TIMEOUT
                )

                # An unknown URL must be submitted before it can be read back.
                if response.status_code == 404:
                    submit_response = self.session.post(
                        f"{self.virustotal_base_url}/urls",
                        headers=headers,
                        data={"url": url},
                        timeout=REQUEST_TIMEOUT,
                    )
                    if submit_response.status_code == 200:
                        logger.info("Submitted %s to VirusTotal for analysis", url)
                        time.sleep(5)  # Give the scanners a moment to run.
                        response = self.session.get(
                            api_url, headers=headers, timeout=REQUEST_TIMEOUT
                        )

                if response.status_code != 200:
                    logger.warning(
                        "VirusTotal request failed with status %s", response.status_code
                    )
                    return enriched

                data = response.json()
                self.api_cache[cache_key] = data

            except requests.RequestException as exc:
                logger.warning("VirusTotal call failed: %s", exc)
                return enriched

        try:
            attributes = data.get("data", {}).get("attributes", {})
            html_content = attributes.get("last_http_response_content", "")

            if html_content:
                enriched.update(extract_html_features(html_content, missing_features))

            if "IsHTTPS" in missing_features:
                last_url = attributes.get("last_final_url") or attributes.get("url", "")
                if last_url:
                    enriched["IsHTTPS"] = 1 if last_url.startswith("https://") else 0

            return enriched

        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning("Could not parse the VirusTotal payload: %s", exc)
            return enriched

    # ------------------------------------------------------------------ #
    # Source 2: WhoisXML
    # ------------------------------------------------------------------ #
    def _try_whoisxml_api(
        self, url: str, missing_features: list[str]
    ) -> dict[str, Any]:
        """Resolve domain-level features through the WhoisXML API."""
        enriched: dict[str, Any] = {}
        if not self.whoisxml_api_key or "DomainAge" not in missing_features:
            return enriched

        domain = urllib.parse.urlparse(url).netloc
        if not domain:
            return enriched

        cache_key = f"whoisxml_{domain}"
        if cache_key in self.api_cache:
            data = self.api_cache[cache_key]
        else:
            try:
                response = self.session.get(
                    self.whoisxml_base_url,
                    params={
                        "apiKey": self.whoisxml_api_key,
                        "domainName": domain,
                        "outputFormat": "JSON",
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                if response.status_code != 200:
                    logger.warning(
                        "WhoisXML request failed with status %s", response.status_code
                    )
                    return enriched
                data = response.json()
                self.api_cache[cache_key] = data
            except requests.RequestException as exc:
                logger.warning("WhoisXML call failed: %s", exc)
                return enriched

        try:
            record = data.get("WhoisRecord", {})
            estimated_age = record.get("estimatedDomainAge")
            if estimated_age is not None:
                # The model treats "recently registered" (< ~6 months) as risky.
                enriched["DomainAge"] = 0 if int(estimated_age) >= 180 else 1
        except (TypeError, ValueError) as exc:
            logger.warning("Could not parse the WhoisXML payload: %s", exc)

        return enriched

    # ------------------------------------------------------------------ #
    # Source 3: urlscan.io
    # ------------------------------------------------------------------ #
    def _try_urlscan_for_html(
        self, url: str, missing_features: list[str]
    ) -> dict[str, Any]:
        """Recover HTML-derived features from a urlscan.io snapshot.

        urlscan exposes historical scans without authentication, so this step
        runs even when no API key is configured.
        """
        enriched: dict[str, Any] = {}
        domain = urllib.parse.urlparse(url).netloc
        if not domain:
            return enriched

        cache_key = f"urlscan_{domain}"
        if cache_key in self.api_cache:
            html_content = self.api_cache[cache_key]
        else:
            try:
                headers = {"Accept": "application/json"}
                if self.urlscan_api_key:
                    headers["API-Key"] = self.urlscan_api_key

                search = self.session.get(
                    f"{self.urlscan_base_url}/search/",
                    params={"q": f"domain:{domain}", "size": 1},
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                )
                if search.status_code != 200:
                    logger.warning(
                        "urlscan search failed with status %s", search.status_code
                    )
                    return enriched

                results = search.json().get("results", [])
                if not results:
                    logger.info("No urlscan snapshot available for %s", domain)
                    return enriched

                dom_url = results[0].get("result", "").replace("/result/", "/dom/")
                if not dom_url:
                    return enriched

                dom_response = self.session.get(dom_url, timeout=REQUEST_TIMEOUT)
                if dom_response.status_code != 200:
                    return enriched

                html_content = dom_response.text
                self.api_cache[cache_key] = html_content

            except requests.RequestException as exc:
                logger.warning("urlscan call failed: %s", exc)
                return enriched

        if html_content:
            enriched.update(extract_html_features(html_content, missing_features))

        return enriched

    # ------------------------------------------------------------------ #
    # Source 4: direct connection
    # ------------------------------------------------------------------ #
    def _get_domain_features(
        self, url: str, missing_features: list[str]
    ) -> dict[str, Any]:
        """Establish transport-level facts by connecting to the host directly."""
        enriched: dict[str, Any] = {}
        if "IsHTTPS" not in missing_features:
            return enriched

        parsed = urllib.parse.urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            enriched["IsHTTPS"] = 0
            return enriched

        # The scheme already answers the question when it is https.
        if parsed.scheme == "https":
            enriched["IsHTTPS"] = 1
            return enriched

        # Otherwise, check whether the host would accept a TLS handshake.
        try:
            context = ssl.create_default_context()
            with socket.create_connection((hostname, 443), timeout=REQUEST_TIMEOUT):
                with context.wrap_socket(
                    socket.socket(), server_hostname=hostname
                ) as sock:
                    sock.settimeout(REQUEST_TIMEOUT)
                    sock.connect((hostname, 443))
                    enriched["IsHTTPS"] = 1
        except (OSError, ssl.SSLError) as exc:
            logger.info("No usable TLS endpoint for %s: %s", hostname, exc)
            enriched["IsHTTPS"] = 0

        return enriched

    # ------------------------------------------------------------------ #
    # Source 5: offline string analysis
    # ------------------------------------------------------------------ #
    def _calculate_url_features(
        self, url: str, missing_features: list[str]
    ) -> dict[str, Any]:
        """Compute the features derivable from the URL string alone."""
        enriched: dict[str, Any] = {}
        extracted = tldextract.extract(url)

        computations = {
            "URLLength": lambda: len(url),
            "NoOfSubDomain": lambda: (
                len(extracted.subdomain.split(".")) if extracted.subdomain else 0
            ),
            "NoOfDots": lambda: url.count("."),
            "NoOfObfuscatedChar": lambda: len(re.findall(r"%[0-9a-fA-F]{2}", url)),
            "NoOfQmark": lambda: url.count("?"),
            "NoOfDigits": lambda: sum(character.isdigit() for character in url),
        }

        for feature in missing_features:
            if feature in computations:
                enriched[feature] = computations[feature]()

        return enriched

    # ------------------------------------------------------------------ #
    # Source 6: defaults
    # ------------------------------------------------------------------ #
    def _generate_default_values(self, missing_features: list[str]) -> dict[str, Any]:
        """Return the neutral fallback value for each requested feature."""
        return {
            feature: self.default_values.get(feature, 0) for feature in missing_features
        }
