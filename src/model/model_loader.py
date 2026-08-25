"""Load the trained LightGBM booster and run predictions with it.

The pickle produced by the training pipeline bundles more than the booster: it
also carries the fitted ``PowerTransformer`` and ``StandardScaler``, the feature
selection mask, and the evaluation metrics. :class:`ModelLoader` restores all of
them and reproduces the exact transformation order used at training time.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_FEATURE_VALUES,
    EXPECTED_FEATURES,
    PREDICTION_THRESHOLD,
)
from src.utils.feature_enrichment import FeatureEnrichment

logger = logging.getLogger(__name__)


class ModelLoader:
    """Restore the trained model bundle and expose a prediction interface."""

    def __init__(self, model_path: str | Path):
        """
        Args:
            model_path: Path to the pickled model bundle.
        """
        self.model_path = Path(model_path)
        self.model = None
        self.power_transformer = None
        self.scaler = None
        self.feature_names: list[str] = []
        self.selected_feature_names: list[str] = []
        self.selected_features_mask = None
        self.metrics: dict[str, Any] = {}

        self.expected_features = list(EXPECTED_FEATURES)
        self.default_values = dict(DEFAULT_FEATURE_VALUES)
        self.feature_enricher = FeatureEnrichment()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load(self) -> bool:
        """Restore the model bundle from disk.

        Returns:
            ``True`` when every essential component was recovered.
        """
        try:
            logger.info("Loading model from: %s", self.model_path)

            with open(self.model_path, "rb") as handle:
                bundle = pickle.load(handle)

            self.model = bundle.get("model")
            self.power_transformer = bundle.get("power_transformer")
            self.scaler = bundle.get("scaler")
            self.selected_features_mask = bundle.get("selected_features_mask")
            self.selected_feature_names = self._as_list(
                bundle.get("selected_feature_names", [])
            )
            self.feature_names = self._as_list(bundle.get("feature_names", []))
            self.metrics = bundle.get("metrics", {})

            if self.model is None or not self.selected_feature_names:
                logger.error("Model bundle is missing essential components")
                return False

            unexpected = set(self.selected_feature_names) - set(self.expected_features)
            if unexpected:
                logger.warning(
                    "Model uses features absent from the expected list: %s", unexpected
                )

            logger.info(
                "Model loaded successfully with %d selected features",
                len(self.selected_feature_names),
            )
            return True

        except (OSError, pickle.UnpicklingError, AttributeError):
            logger.exception("Failed to load the model from %s", self.model_path)
            return False

    def export_metrics(self, destination: str | Path) -> bool:
        """Write the bundled evaluation metrics to a JSON file.

        Kept separate from :meth:`load` so that reading the model never mutates
        the filesystem. The file is written atomically: a partially serialised
        payload can never replace a valid one.

        Args:
            destination: Path of the JSON file to write.

        Returns:
            ``True`` when the metrics were written successfully.
        """
        destination = Path(destination)
        temporary = destination.with_suffix(".json.tmp")

        try:
            payload = {
                key: self._jsonable(value) for key, value in self.metrics.items()
            }
            payload["selected_feature_names"] = self._as_list(
                self.selected_feature_names
            )

            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            temporary.replace(destination)

            logger.info("Model metrics written to: %s", destination)
            return True

        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Could not export the model metrics: %s", exc)
            temporary.unlink(missing_ok=True)
            return False

    # ------------------------------------------------------------------ #
    # Feature preparation
    # ------------------------------------------------------------------ #
    def get_required_features(self) -> list[str]:
        """Return the ordered list of features the booster expects."""
        return self._as_list(self.selected_feature_names)

    def ensure_feature_consistency(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Guarantee the frame holds every required feature, in the right order.

        Missing values are recovered through :class:`FeatureEnrichment` first and
        replaced with neutral defaults only as a last resort.

        Args:
            features_df: Frame of locally extracted features.

        Returns:
            A frame containing exactly the required feature columns, in order,
            plus the original ``url`` column when it was present.
        """
        required = self.get_required_features()
        features_df = features_df.copy()

        missing = [name for name in required if name not in features_df.columns]
        if missing:
            logger.warning("Missing features: %s", missing)

            if "url" in features_df.columns and not features_df.empty:
                url = features_df["url"].iloc[0]
                context = features_df.to_dict("records")[0]
                enriched = self.feature_enricher.enrich_features(url, context, missing)
                for feature, value in enriched.items():
                    features_df[feature] = value

            still_missing = [
                name for name in required if name not in features_df.columns
            ]
            if still_missing:
                logger.warning(
                    "Using default values after enrichment for: %s", still_missing
                )
                for feature in still_missing:
                    features_df[feature] = self.default_values.get(feature, 0)

        for feature in required:
            column = features_df[feature]

            if column.dtype == "object":
                logger.warning("Coercing non-numeric column %s", feature)
                column = pd.to_numeric(column, errors="coerce")

            if column.isnull().any():
                logger.warning("Filling missing values in %s", feature)
                column = column.fillna(self.default_values.get(feature, 0))

            features_df[feature] = column

        result = features_df[required].copy()
        if "url" in features_df.columns:
            result["url"] = features_df["url"]

        return result

    def preprocess_features(self, features_df: pd.DataFrame) -> np.ndarray | None:
        """Apply the training-time transformation chain to ``features_df``.

        The transformers were fitted on the full feature space, so the selected
        columns are first widened back to that space, transformed, then narrowed
        again through the selection mask.

        Args:
            features_df: Frame holding at least every selected feature.

        Returns:
            The transformed matrix, or ``None`` if a required column is absent.
        """
        missing = [
            name for name in self.selected_feature_names if name not in features_df
        ]
        if missing:
            logger.error("Cannot preprocess, missing columns: %s", missing)
            return None

        matrix = features_df[self.selected_feature_names].to_numpy(dtype=float)
        logger.info("Feature matrix before preprocessing: %s", matrix.shape)

        if self.feature_names:
            matrix = self._expand_to_training_space(matrix)

        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        if self.power_transformer is not None:
            matrix = self._apply_transformer(
                self.power_transformer, matrix, "PowerTransformer"
            )

        if self.scaler is not None:
            matrix = self._apply_transformer(self.scaler, matrix, "StandardScaler")

        if (
            self.selected_features_mask is not None
            and len(self.selected_features_mask) == matrix.shape[1]
        ):
            matrix = matrix[:, self.selected_features_mask]
            logger.info("Feature matrix after selection mask: %s", matrix.shape)

        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict(self, features_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Score a frame of extracted features.

        Args:
            features_df: Features for one or more URLs.

        Returns:
            A ``(probabilities, labels)`` pair where a label of 1 means phishing.
            On failure the caller receives a maximally uncertain 0.5 probability
            rather than an exception, so the UI can still render a result.
        """
        undecided = (np.array([0.5]), np.array([0]))

        try:
            prepared = self.ensure_feature_consistency(features_df)
            matrix = self.preprocess_features(prepared)

            if matrix is None:
                logger.error("Preprocessing failed, returning an undecided result")
                return undecided

            raw_scores = np.asarray(self.model.predict(matrix)).ravel()

            # A Booster trained with a binary objective already returns
            # probabilities; anything outside [0, 1] is a raw margin.
            if raw_scores.min() < 0.0 or raw_scores.max() > 1.0:
                probabilities = 1.0 / (1.0 + np.exp(-raw_scores))
            else:
                probabilities = raw_scores

            labels = (probabilities >= PREDICTION_THRESHOLD).astype(int)
            logger.info("Probabilities: %s, labels: %s", probabilities, labels)

            return probabilities, labels

        except (ValueError, TypeError, AttributeError):
            logger.exception("Prediction failed")
            return undecided

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def get_model_components(self) -> dict[str, Any]:
        """Return the metrics and feature names used by the reporting views."""
        return {
            "metrics": self.metrics,
            "selected_feature_names": self.get_required_features(),
        }

    def get_feature_importance(self) -> tuple[list[str], list[float]]:
        """Return feature names paired with their importance scores.

        Falls back to a uniform distribution when the bundle carries no
        importance information, so callers can always plot something.
        """
        names = self.get_required_features()

        importances = None
        if hasattr(self.model, "feature_importance"):  # LightGBM Booster
            importances = self.model.feature_importance()
        elif hasattr(self.model, "feature_importances_"):  # scikit-learn API
            importances = self.model.feature_importances_
        elif "feature_importance" in self.metrics:
            importances = self.metrics["feature_importance"]

        if importances is None or len(importances) == 0:
            logger.warning("No feature importance available, using a uniform split")
            uniform = 1.0 / len(names) if names else 0.0
            return names, [uniform] * len(names)

        return names, self._as_list(importances)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _expand_to_training_space(self, matrix: np.ndarray) -> np.ndarray:
        """Widen a selected-feature matrix back to the full training layout."""
        logger.info("Expanding features to the transformer's training layout")
        expanded = np.zeros((matrix.shape[0], len(self.feature_names)))
        selected = list(self.selected_feature_names)

        for target_index, feature in enumerate(self.feature_names):
            if feature in selected:
                expanded[:, target_index] = matrix[:, selected.index(feature)]

        logger.info("Feature matrix after expansion: %s", expanded.shape)
        return expanded

    @staticmethod
    def _apply_transformer(transformer, matrix: np.ndarray, label: str) -> np.ndarray:
        """Apply a fitted transformer, degrading gracefully if it rejects input."""
        try:
            logger.info("Applying %s", label)
            return transformer.transform(matrix)
        except (ValueError, AttributeError) as exc:
            logger.error("%s failed, continuing untransformed: %s", label, exc)
            return matrix

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        """Normalise numpy arrays, pandas indexes and sequences to a list."""
        if value is None:
            return []
        if hasattr(value, "tolist"):
            return value.tolist()
        return list(value)

    @classmethod
    def _jsonable(cls, value: Any) -> Any:
        """Convert a metric value into something ``json.dump`` accepts.

        The bundle mixes plain scalars with numpy arrays and pandas indexes;
        serialising the latter directly is what produced a truncated metrics
        file in earlier versions.
        """
        if isinstance(value, (str, bool, int, float)) or value is None:
            return value
        if isinstance(value, np.generic):
            return value.item()
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, dict):
            return {str(key): cls._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._jsonable(item) for item in value]
        return str(value)
