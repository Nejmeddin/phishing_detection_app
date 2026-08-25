"""Tests for configuration integrity.

The secret-scanning test exists because this repository previously shipped live
API keys in source. It fails the build if a credential is ever hardcoded again.
"""

import re
from pathlib import Path

import pytest

from src import config

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Shapes of the credentials this project uses. Each pattern targets a literal
# assignment, so reading a key from the environment never trips them.
SECRET_PATTERNS = [
    ("VirusTotal API key", re.compile(r"[\"'][0-9a-f]{64}[\"']")),
    ("WhoisXML API key", re.compile(r"[\"']at_[A-Za-z0-9]{20,}[\"']")),
    ("urlscan API key", re.compile(r"[\"'][0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-")),
]


def _source_files():
    """Every tracked Python file, excluding caches and virtual environments."""
    skip = {"__pycache__", ".venv", "venv", "evenv", ".git"}
    for path in PROJECT_ROOT.rglob("*.py"):
        if not skip.intersection(path.parts):
            yield path


class TestNoHardcodedSecrets:
    @pytest.mark.parametrize("label,pattern", SECRET_PATTERNS)
    def test_no_credential_literals_in_source(self, label, pattern):
        offenders = [
            str(path.relative_to(PROJECT_ROOT))
            for path in _source_files()
            if pattern.search(path.read_text(encoding="utf-8"))
        ]
        assert not offenders, f"{label} appears hardcoded in: {offenders}"

    def test_keys_default_to_empty(self, monkeypatch):
        """Absent an environment variable, a key must be empty, not a literal."""
        for name in ("VIRUSTOTAL_API_KEY", "URLSCAN_API_KEY", "WHOISXML_API_KEY"):
            monkeypatch.delenv(name, raising=False)

        import importlib

        reloaded = importlib.reload(config)
        assert reloaded.VIRUSTOTAL_API_KEY == ""
        assert reloaded.URLSCAN_API_KEY == ""
        assert reloaded.WHOISXML_API_KEY == ""

    def test_env_example_ships_no_values(self):
        example = PROJECT_ROOT / ".env.example"
        assert example.exists(), ".env.example should document the required keys"

        for line in example.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("#") or "=" not in line:
                continue
            _, _, value = line.partition("=")
            assert value.strip() == "", f"{line} should not carry a real value"


class TestFeatureDefinitions:
    def test_expected_features_are_unique(self):
        assert len(config.EXPECTED_FEATURES) == len(set(config.EXPECTED_FEATURES))

    def test_defaults_cover_every_expected_feature(self):
        assert set(config.DEFAULT_FEATURE_VALUES) == set(config.EXPECTED_FEATURES)

    def test_html_and_url_features_are_disjoint(self):
        assert not config.HTML_FEATURES & config.URL_FEATURES

    def test_feature_groups_are_subsets_of_the_expected_set(self):
        expected = set(config.EXPECTED_FEATURES)
        assert config.HTML_FEATURES.issubset(expected)
        assert config.URL_FEATURES.issubset(expected)

    def test_every_expected_feature_has_a_display_name(self):
        missing = set(config.EXPECTED_FEATURES) - set(config.FEATURE_DISPLAY_NAMES)
        assert not missing, f"features without a display name: {missing}"

    def test_every_expected_feature_has_an_explanation(self):
        missing = set(config.EXPECTED_FEATURES) - set(config.FEATURE_EXPLANATIONS)
        assert not missing, f"features without an explanation: {missing}"


class TestPaths:
    def test_paths_are_anchored_to_the_project_root(self):
        assert config.BASE_DIR == PROJECT_ROOT
        assert config.RAW_DATA_DIR.parent == config.DATA_DIR
        assert config.PROCESSED_DATA_DIR.parent == config.DATA_DIR
