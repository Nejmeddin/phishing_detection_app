"""Tests for URL feature extraction.

Network access is always mocked: extraction must be deterministic, and the
failure path matters as much as the success path.
"""

from unittest.mock import patch

import pytest
import requests

from src.preprocessing.feature_extractor import EnhancedFeatureExtractor


@pytest.fixture
def extractor():
    """An extractor whose WHOIS lookups are disabled."""
    with patch.object(
        EnhancedFeatureExtractor, "_get_domain_age_flag", return_value=None
    ):
        yield EnhancedFeatureExtractor()


def _unreachable(*_args, **_kwargs):
    raise requests.ConnectionError("offline")


class TestLexicalFeatures:
    """Features derived from the URL string alone."""

    def test_https_is_detected(self, extractor):
        with patch.object(extractor.session, "get", _unreachable):
            features = extractor.extract_features("https://www.example.com")
        assert features["IsHTTPS"] == 1

    def test_plain_http_is_detected(self, extractor):
        with patch.object(extractor.session, "get", _unreachable):
            features = extractor.extract_features("http://www.example.com")
        assert features["IsHTTPS"] == 0

    def test_scheme_is_added_when_missing(self, extractor):
        with patch.object(extractor.session, "get", _unreachable):
            features = extractor.extract_features("example.com")
        assert features["URLLength"] == len("http://example.com")

    def test_counts_structural_characters(self, extractor):
        url = "http://a.b.example.com/p?x=1&y=22"
        with patch.object(extractor.session, "get", _unreachable):
            features = extractor.extract_features(url)

        assert features["NoOfSubDomain"] == 2
        assert features["NoOfDots"] == url.count(".")
        assert features["NoOfQmark"] == 1
        assert features["NoOfDigits"] == 3

    def test_detects_percent_encoding(self, extractor):
        with patch.object(extractor.session, "get", _unreachable):
            features = extractor.extract_features(
                "http://example.com/%63%6C%69%65%6E%74"
            )
        assert features["NoOfObfuscatedChar"] == 6

    def test_detects_url_shorteners(self, extractor):
        with patch.object(extractor.session, "get", _unreachable):
            shortened = extractor.extract_features("https://bit.ly/3rGH7q")
            plain = extractor.extract_features("https://example.com/page")

        assert shortened["IsTinyURL"] == 1
        assert plain["IsTinyURL"] == 0

    def test_detects_trust_baiting_keywords_in_domain(self, extractor):
        with patch.object(extractor.session, "get", _unreachable):
            baited = extractor.extract_features("http://secure-login.example.com")
            neutral = extractor.extract_features("http://weather.example.com")

        assert baited["HasSuspiciousKeyword"] == 1
        assert neutral["HasSuspiciousKeyword"] == 0


class TestContentFeatures:
    """Features that require fetching the page."""

    def test_parses_a_reachable_page(self, extractor, legitimate_html):
        class Response:
            text = legitimate_html

        with patch.object(extractor.session, "get", return_value=Response()):
            features = extractor.extract_features("https://example.com")

        assert features["HasTitle"] == 1
        assert features["NoOfImage"] == 2

    def test_unreachable_page_falls_back_to_defaults(self, extractor):
        with patch.object(extractor.session, "get", _unreachable):
            features = extractor.extract_features("https://unreachable.example")

        assert features["HasTitle"] == 0
        assert features["NoOfHyperlink"] == 0

    def test_unreachable_page_still_yields_every_feature(self, extractor):
        with patch.object(extractor.session, "get", _unreachable):
            features = extractor.extract_features("https://unreachable.example")

        missing = set(extractor.expected_features) - set(features)
        assert not missing, f"extraction left features unset: {missing}"

    def test_the_original_url_is_preserved(self, extractor):
        with patch.object(extractor.session, "get", _unreachable):
            features = extractor.extract_features("example.com/path")
        assert features["url"] == "example.com/path"
