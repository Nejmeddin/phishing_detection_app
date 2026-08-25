"""Tests for the model loader.

Tests that need the trained bundle are skipped when it is absent, so the suite
still runs on a fresh clone that has not downloaded the model.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.config import EXPECTED_FEATURES, MODEL_PATH
from src.model.model_loader import ModelLoader

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="trained model bundle is not available"
)


@pytest.fixture(scope="module")
def loader():
    """A loaded model, shared across the module."""
    instance = ModelLoader(MODEL_PATH)
    assert instance.load(), "the bundled model failed to load"
    return instance


def _feature_row(**overrides) -> pd.DataFrame:
    """Build a single-row frame covering every expected feature."""
    row = dict.fromkeys(EXPECTED_FEATURES, 0)
    row.update(overrides)
    return pd.DataFrame([row])


class TestLoading:
    def test_reports_success(self, loader):
        assert loader.model is not None

    def test_exposes_the_selected_features(self, loader):
        required = loader.get_required_features()
        assert required
        assert set(required).issubset(EXPECTED_FEATURES)

    def test_missing_file_fails_without_raising(self, tmp_path):
        assert ModelLoader(tmp_path / "absent.pkl").load() is False

    def test_loading_does_not_write_to_disk(self, tmp_path):
        """Reading a model must not have filesystem side effects."""
        before = set(MODEL_PATH.parent.iterdir())
        ModelLoader(MODEL_PATH).load()
        assert set(MODEL_PATH.parent.iterdir()) == before


class TestFeatureConsistency:
    def test_returns_columns_in_the_models_order(self, loader):
        result = loader.ensure_feature_consistency(_feature_row())
        expected = loader.get_required_features()
        assert list(result.columns)[: len(expected)] == expected

    def test_fills_missing_columns(self, loader):
        partial = _feature_row().drop(columns=["NoOfJS"])
        result = loader.ensure_feature_consistency(partial)
        assert "NoOfJS" in result.columns

    def test_replaces_nan_values(self, loader):
        frame = _feature_row(NoOfJS=np.nan)
        result = loader.ensure_feature_consistency(frame)
        assert not result[loader.get_required_features()].isnull().any().any()

    def test_coerces_non_numeric_values(self, loader):
        frame = _feature_row(NoOfJS="not a number")
        result = loader.ensure_feature_consistency(frame)
        assert result["NoOfJS"].iloc[0] == 0

    def test_preserves_the_url_column(self, loader):
        frame = _feature_row()
        frame["url"] = "https://example.com"
        assert "url" in loader.ensure_feature_consistency(frame).columns


class TestPrediction:
    def test_returns_a_probability_and_a_label(self, loader):
        probabilities, labels = loader.predict(_feature_row())

        assert len(probabilities) == 1
        assert 0.0 <= probabilities[0] <= 1.0
        assert labels[0] in (0, 1)

    def test_separates_obvious_cases(self, loader):
        """A page with every trust marker should score below a bare login form."""
        legitimate = _feature_row(
            IsHTTPS=1,
            URLLength=23,
            NoOfSubDomain=1,
            NoOfDots=2,
            LineLength=900,
            HasTitle=1,
            HasMeta=1,
            HasFavicon=1,
            HasCopyright=1,
            HasSocialNetworking=1,
            HasSubmitButton=1,
            NoOfImage=40,
            NoOfJS=25,
            NoOfCSS=10,
            NoOfHyperlink=180,
        )
        suspicious = _feature_row(
            IsHTTPS=0,
            URLLength=120,
            NoOfSubDomain=4,
            NoOfDots=7,
            NoOfObfuscatedChar=6,
            NoOfDigits=15,
            LineLength=40,
            HasPasswordField=1,
            HasSubmitButton=1,
            NoOfiFrame=2,
            NoOfHyperlink=1,
            NoOfJS=2,
        )

        legitimate_score = loader.predict(legitimate)[0][0]
        suspicious_score = loader.predict(suspicious)[0][0]

        assert legitimate_score < suspicious_score

    def test_malformed_input_yields_an_undecided_result(self, loader):
        probabilities, labels = loader.predict(pd.DataFrame([{"url": "x"}]))
        assert 0.0 <= probabilities[0] <= 1.0
        assert labels[0] in (0, 1)


class TestMetricsExport:
    def test_writes_valid_json(self, loader, tmp_path):
        destination = tmp_path / "metrics.json"
        assert loader.export_metrics(destination) is True

        payload = json.loads(destination.read_text(encoding="utf-8"))
        assert "accuracy" in payload
        assert isinstance(payload["confusion_matrix"], list)

    def test_leaves_no_temporary_file_behind(self, loader, tmp_path):
        destination = tmp_path / "metrics.json"
        loader.export_metrics(destination)
        assert list(tmp_path.iterdir()) == [destination]

    def test_serialises_numpy_and_pandas_types(self, loader):
        assert ModelLoader._jsonable(np.int64(3)) == 3
        assert ModelLoader._jsonable(np.array([1, 2])) == [1, 2]
        assert ModelLoader._jsonable(pd.Index(["a", "b"])) == ["a", "b"]

    def test_feature_importance_matches_the_feature_count(self, loader):
        names, importances = loader.get_feature_importance()
        assert len(names) == len(importances)
