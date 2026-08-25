"""Tests for the shared HTML feature parser."""

from src.config import HTML_FEATURES
from src.preprocessing.html_features import extract_html_features


def test_extracts_every_html_feature_by_default(legitimate_html):
    features = extract_html_features(legitimate_html)
    assert set(features) == set(HTML_FEATURES)


def test_recognises_markers_of_a_legitimate_page(legitimate_html):
    features = extract_html_features(legitimate_html)

    assert features["HasTitle"] == 1
    assert features["HasMeta"] == 1
    assert features["HasFavicon"] == 1
    assert features["HasCopyright"] == 1
    assert features["HasSocialNetworking"] == 1
    assert features["HasPasswordField"] == 0
    assert features["NoOfImage"] == 2
    assert features["NoOfCSS"] == 2  # one stylesheet link plus one inline block
    assert features["NoOfHyperlink"] == 2


def test_recognises_markers_of_a_phishing_page(phishing_html):
    features = extract_html_features(phishing_html)

    assert features["HasTitle"] == 0
    assert features["HasFavicon"] == 0
    assert features["HasPasswordField"] == 1
    assert features["HasSubmitButton"] == 1
    assert features["NoOfiFrame"] == 1
    assert features["NoOfURLRedirect"] == 2


def test_only_requested_features_are_computed(legitimate_html):
    features = extract_html_features(legitimate_html, ["HasTitle", "NoOfImage"])
    assert set(features) == {"HasTitle", "NoOfImage"}


def test_unknown_feature_names_are_ignored(legitimate_html):
    assert extract_html_features(legitimate_html, ["NotAFeature"]) == {}


def test_empty_document_yields_zeros():
    features = extract_html_features("")
    assert features["HasTitle"] == 0
    assert features["NoOfHyperlink"] == 0
