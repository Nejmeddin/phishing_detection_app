"""Tests for the feature enrichment fallbacks.

These cover the contract that matters to the model: whatever happens to the
external services, ``enrich_features`` returns a value for every feature it was
asked about.
"""

from unittest.mock import patch

import pytest
import requests

from src.utils.feature_enrichment import FeatureEnrichment


@pytest.fixture
def enricher():
    """An enricher with no API keys, i.e. the fully offline configuration."""
    instance = FeatureEnrichment()
    instance.virustotal_api_key = ""
    instance.urlscan_api_key = ""
    instance.whoisxml_api_key = ""
    return instance


def _unreachable(*_args, **_kwargs):
    raise requests.ConnectionError("offline")


class TestPublicContract:
    def test_every_requested_feature_is_returned(self, enricher):
        requested = ["HasTitle", "NoOfJS", "URLLength", "IsHTTPS"]
        with patch.object(enricher.session, "get", _unreachable):
            result = enricher.enrich_features("https://example.com", {}, requested)

        assert set(requested).issubset(result)

    def test_no_request_means_no_result(self, enricher):
        assert enricher.enrich_features("https://example.com", {}, []) == {}

    @pytest.mark.parametrize("bad_url", ["", None, 12345])
    def test_invalid_urls_degrade_to_defaults(self, enricher, bad_url):
        result = enricher.enrich_features(bad_url, {}, ["HasTitle", "NoOfJS"])
        assert result == {"HasTitle": 0, "NoOfJS": 0}

    def test_network_failure_still_produces_a_full_vector(self, enricher):
        requested = ["HasTitle", "HasMeta", "NoOfImage", "NoOfJS"]
        with patch.object(enricher.session, "get", _unreachable):
            result = enricher.enrich_features("https://example.com", {}, requested)

        assert set(result) == set(requested)
        assert all(value == 0 for value in result.values())


class TestOfflineComputation:
    """The URL-only path, which never touches the network."""

    def test_computes_lexical_features(self, enricher):
        result = enricher._calculate_url_features(
            "http://a.b.example.com/x?y=1%2F2",
            ["URLLength", "NoOfDots", "NoOfSubDomain", "NoOfDigits", "NoOfQmark"],
        )

        assert result["NoOfDots"] == 3
        assert result["NoOfSubDomain"] == 2
        assert result["NoOfQmark"] == 1
        assert result["NoOfDigits"] == 3

    def test_lexical_features_are_preferred_over_defaults(self, enricher):
        with patch.object(enricher.session, "get", _unreachable):
            result = enricher.enrich_features(
                "http://example.com/verylongpath", {}, ["URLLength"]
            )

        assert result["URLLength"] == len("http://example.com/verylongpath")

    def test_defaults_are_neutral(self, enricher):
        assert enricher._generate_default_values(["HasTitle", "NoOfJS"]) == {
            "HasTitle": 0,
            "NoOfJS": 0,
        }

    def test_https_scheme_is_recognised_without_a_handshake(self, enricher):
        assert enricher._get_domain_features("https://example.com", ["IsHTTPS"]) == {
            "IsHTTPS": 1
        }


class TestDisabledServices:
    """Without credentials, the paid services must be skipped, not attempted."""

    def test_virustotal_is_skipped_without_a_key(self, enricher):
        assert enricher._try_virustotal_api("https://example.com", ["HasTitle"]) == {}

    def test_whoisxml_is_skipped_without_a_key(self, enricher):
        assert enricher._try_whoisxml_api("https://example.com", ["DomainAge"]) == {}
