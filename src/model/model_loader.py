"""
Module pour charger et utiliser le modèle de détection de phishing
"""

import os
import pickle
import logging
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Classe pour charger et utiliser le modèle LightGBM pour la détection de phishing
    """

    def __init__(self, model_path: str):
        """
        Initialise le chargeur de modèle.

        Args:
            model_path: Chemin vers le fichier pickle du modèle
        """
        self.model_path = model_path
        self.model = None
        self.power_transformer = None
        self.scaler = None
        self.feature_names = []
        self.selected_feature_names = []
        self.selected_features_mask = None
        self.metrics = {}
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

        # Importer l'enrichisseur de caractéristiques
        from src.utils.feature_enrichment import FeatureEnrichment

        self.feature_enricher = FeatureEnrichment()

    def load(self) -> bool:
        """
        Charge le modèle et ses composants à partir du fichier pickle.

        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        try:
            logger.info(f"Chargement du modèle depuis: {self.model_path}")

            with open(self.model_path, "rb") as file:
                model_data = pickle.load(file)

            # Extraire les composants du modèle
            self.model = model_data.get("model")
            self.power_transformer = model_data.get("power_transformer")
            self.scaler = model_data.get("scaler")
            self.selected_features_mask = model_data.get("selected_features_mask")
            self.selected_feature_names = model_data.get("selected_feature_names", [])
            self.feature_names = model_data.get("feature_names", [])
            self.metrics = model_data.get("metrics", {})

            # Vérifier que les composants essentiels existent
            if (
                self.model is None
                or self.selected_feature_names is None
                or len(self.selected_feature_names) == 0
            ):
                logger.error("Composants essentiels du modèle manquants")
                return False

            # Vérifier la cohérence avec nos attentes
            missing_expected = set(self.selected_feature_names) - set(
                self.expected_features
            )
            if missing_expected:
                logger.warning(
                    f"Certaines caractéristiques du modèle ne sont pas dans notre liste attendue: {missing_expected}"
                )

            # Sauvegarder les métriques du modèle dans un fichier JSON pour faciliter le débogage
            metrics_file = os.path.join(
                os.path.dirname(self.model_path), "model_metrics.json"
            )
            try:
                with open(metrics_file, "w") as f:
                    # Convertir les arrays numpy en listes pour JSON
                    metrics_json = {}
                    for key, value in self.metrics.items():
                        if isinstance(value, np.ndarray):
                            metrics_json[key] = value.tolist()
                        else:
                            metrics_json[key] = value

                    # Ajouter les noms des caractéristiques
                    metrics_json["selected_feature_names"] = self.selected_feature_names
                    if isinstance(self.selected_feature_names, np.ndarray):
                        metrics_json["selected_feature_names"] = (
                            self.selected_feature_names.tolist()
                        )

                    json.dump(metrics_json, f, indent=2)
                    logger.info(
                        f"Métriques du modèle sauvegardées dans: {metrics_file}"
                    )
            except Exception as e:
                logger.warning(f"Impossible de sauvegarder les métriques: {str(e)}")

            logger.info(
                f"Modèle chargé avec succès. {len(self.selected_feature_names)} caractéristiques sélectionnées"
            )
            return True

        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    def get_required_features(self) -> List[str]:
        """
        Retourne la liste des caractéristiques requises par le modèle.

        Returns:
            List[str]: Liste des noms de caractéristiques
        """
        # Convertir le pandas.Index en liste Python si nécessaire
        if self.selected_feature_names is not None and hasattr(
            self.selected_feature_names, "tolist"
        ):
            return self.selected_feature_names.tolist()
        elif self.selected_feature_names is not None:
            return list(self.selected_feature_names)
        else:
            return []

    def ensure_feature_consistency(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        S'assure que toutes les caractéristiques nécessaires sont présentes et dans le bon ordre.
        Utilise l'enrichisseur de caractéristiques pour compléter les valeurs manquantes.

        Args:
            features_df: DataFrame contenant les caractéristiques extraites

        Returns:
            pd.DataFrame: DataFrame avec toutes les caractéristiques requises
        """
        required_features = self.get_required_features()

        # Log initial pour le débogage
        logger.info(f"Features DataFrame initial: {features_df.columns.tolist()}")
        logger.info(f"Features requises par le modèle: {required_features}")

        # Vérifier les caractéristiques manquantes
        missing_features = [
            f for f in required_features if f not in features_df.columns
        ]

        if missing_features:
            logger.warning(f"Caractéristiques manquantes: {missing_features}")

            # Utiliser l'enrichisseur pour obtenir les caractéristiques manquantes
            if "url" in features_df.columns:
                url = features_df["url"].iloc[0]
                features_dict = features_df.to_dict("records")[0]

                # Tentative d'enrichissement
                enriched_features = self.feature_enricher.enrich_features(
                    url, features_dict, missing_features
                )

                # Ajouter les caractéristiques enrichies au DataFrame
                for feature, value in enriched_features.items():
                    features_df[feature] = value

            # Vérifier s'il reste des caractéristiques manquantes après enrichissement
            still_missing = [
                f for f in required_features if f not in features_df.columns
            ]
            if still_missing:
                logger.warning(
                    f"Caractéristiques toujours manquantes après enrichissement: {still_missing}"
                )
                for feature in still_missing:
                    features_df[feature] = self.default_values.get(feature, 0)

        # Validation de l'intégrité des données
        for feature in required_features:
            # Vérifier les valeurs manquantes ou NaN
            if features_df[feature].isnull().any():
                logger.warning(
                    f"Valeurs NaN détectées pour {feature}, remplacement par valeur par défaut"
                )
                features_df[feature] = features_df[feature].fillna(
                    self.default_values.get(feature, 0)
                )

            # Vérifier les types de données et convertir si nécessaire
            if features_df[feature].dtype == "object":
                logger.warning(
                    f"Type de données incorrect pour {feature}, conversion en numérique"
                )
                try:
                    features_df[feature] = pd.to_numeric(
                        features_df[feature], errors="coerce"
                    )
                    # Remplacer les NaN après conversion
                    features_df[feature] = features_df[feature].fillna(
                        self.default_values.get(feature, 0)
                    )
                except:
                    features_df[feature] = self.default_values.get(feature, 0)

        # S'assurer que les colonnes sont dans le bon ordre
        result_df = features_df[required_features].copy()

        # Conserver la colonne url si elle existe
        if "url" in features_df.columns:
            result_df["url"] = features_df["url"]

        # Log final pour le débogage
        logger.info(f"Features DataFrame final: {result_df.columns.tolist()}")

        return result_df

    def preprocess_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Prétraite les caractéristiques avec le PowerTransformer et le StandardScaler.

        Args:
            features_df: DataFrame contenant les caractéristiques

        Returns:
            np.ndarray: Caractéristiques prétraitées
        """
        try:
            # Vérifier que les caractéristiques requises sont présentes
            if not all(f in features_df.columns for f in self.selected_feature_names):
                missing = [
                    f
                    for f in self.selected_feature_names
                    if f not in features_df.columns
                ]
                logger.error(
                    f"Caractéristiques manquantes pour le prétraitement: {missing}"
                )
                return None

            # Récupérer seulement les caractéristiques sélectionnées
            X = features_df[self.selected_feature_names].values

            # Log pour le débogage
            logger.info(f"Forme des données avant prétraitement: {X.shape}")

            # Créer un DataFrame avec toutes les caractéristiques originales attendues par le PowerTransformer
            if self.feature_names is not None and len(self.feature_names) > 0:
                logger.info(
                    f"Adaptation des features pour correspondre au format attendu par le PowerTransformer"
                )
                # Initialiser un array de zéros avec les bonnes dimensions
                full_X = np.zeros((X.shape[0], len(self.feature_names)))

                # Remplir avec les valeurs disponibles
                for i, feature in enumerate(self.feature_names):
                    if feature in self.selected_feature_names:
                        # Trouver l'index de cette caractéristique dans selected_feature_names
                        idx = list(self.selected_feature_names).index(feature)
                        # Copier la valeur
                        full_X[:, i] = X[:, idx]

                # Utiliser ce tableau complet pour le prétraitement
                X = full_X
                logger.info(f"Forme des données après adaptation: {X.shape}")

            # Vérifier les valeurs invalides
            if np.isnan(X).any() or np.isinf(X).any():
                logger.warning(
                    "Valeurs NaN ou Inf détectées dans les données. Remplacement par zéros."
                )
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Appliquer PowerTransformer si disponible
            if self.power_transformer:
                logger.info("Application du PowerTransformer")
                try:
                    X = self.power_transformer.transform(X)
                except Exception as e:
                    logger.error(f"Erreur avec PowerTransformer: {str(e)}")
                    # Continuer sans cette transformation
                    pass

            # Appliquer StandardScaler si disponible
            if self.scaler:
                logger.info("Application du StandardScaler")
                try:
                    X = self.scaler.transform(X)
                except Exception as e:
                    logger.error(f"Erreur avec StandardScaler: {str(e)}")
                    # Continuer sans cette transformation
                    pass

            # Si nous avions étendu X, récupérer seulement les colonnes sélectionnées
            if (
                self.selected_features_mask is not None
                and len(self.selected_features_mask) == X.shape[1]
            ):
                logger.info(f"Application du masque de sélection de features")
                X = X[:, self.selected_features_mask]
                logger.info(f"Forme des données après masque: {X.shape}")

            # Vérification finale
            if np.isnan(X).any() or np.isinf(X).any():
                logger.warning(
                    "Valeurs NaN ou Inf détectées après prétraitement. Remplacement par zéros."
                )
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            return X

        except Exception as e:
            logger.error(f"Erreur lors du prétraitement: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    def predict(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Effectue une prédiction avec le modèle.

        Args:
            features_df: DataFrame contenant les caractéristiques

        Returns:
            Tuple[np.ndarray, np.ndarray]: (probabilités, prédictions)
        """
        try:
            # Log pour le suivi
            logger.info(
                f"Prédiction pour les caractéristiques: {features_df.columns.tolist()}"
            )

            # S'assurer que les features sont cohérentes
            features_df = self.ensure_feature_consistency(features_df)

            # Prétraitement des caractéristiques
            X = self.preprocess_features(features_df)

            if X is None:
                logger.error("Échec du prétraitement des caractéristiques")
                return np.array([0.5]), np.array([0])

            # Prédiction des probabilités
            logger.info(f"Données pour prédiction de forme: {X.shape}")
            raw_probas = self.model.predict(X)
            logger.info(f"Prédiction brute obtenue: {raw_probas}")

            # Si le modèle retourne directement les probabilités
            if (
                raw_probas.ndim == 1
                and (0 <= raw_probas.min() <= 1)
                and (0 <= raw_probas.max() <= 1)
            ):
                probas = raw_probas
            else:
                # Conversion des scores en probabilités si nécessaire
                probas = 1 / (1 + np.exp(-raw_probas))

            # Prédiction des classes (0 = légitime, 1 = phishing)
            predictions = (probas >= 0.5).astype(int)

            logger.info(f"Probabilité calculée: {probas}, Prédiction: {predictions}")

            return probas, predictions

        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            return np.array([0.5]), np.array([0])

    def get_model_components(self) -> Dict[str, Any]:
        """
        Retourne les composants du modèle pour analyse et visualisation.

        Returns:
            Dict: Dictionnaire contenant les composants du modèle
        """
        return {
            "metrics": self.metrics,
            "selected_feature_names": self.selected_feature_names,
        }

    def get_feature_importance(self) -> Tuple[List[str], List[float]]:
        """
        Retourne l'importance des caractéristiques.

        Returns:
            Tuple[List[str], List[float]]: (noms des caractéristiques, scores d'importance)
        """
        if (
            not hasattr(self.model, "feature_importances_")
            and "feature_importance" not in self.metrics
        ):
            logger.warning(
                "Aucune information d'importance des caractéristiques disponible"
            )
            # Retourner des valeurs par défaut
            equal_importance = [1.0 / len(self.selected_feature_names)] * len(
                self.selected_feature_names
            )
            return list(self.selected_feature_names), equal_importance

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        else:
            importances = self.metrics.get("feature_importance", [])

        # Convertir en liste Python si nécessaire
        if isinstance(importances, np.ndarray):
            importances = importances.tolist()
        if isinstance(self.selected_feature_names, np.ndarray):
            feature_names = self.selected_feature_names.tolist()
        else:
            feature_names = list(self.selected_feature_names)

        return feature_names, importances
