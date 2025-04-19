"""
Script de test pour valider l'extraction et l'enrichissement des caractéristiques d'URL
"""

import os
import sys
import pandas as pd
import logging
import time
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("test_features.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Ajout du répertoire parent au chemin d'importation
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

# Import des modules à tester
from src.preprocessing.feature_extractor import WebFeatureExtractor
from src.utils.feature_enrichment import FeatureEnrichment
from src.model.model_loader import ModelLoader


def test_url(url, features_expected, model_loader=None):
    """
    Teste l'extraction de caractéristiques pour une URL donnée

    Args:
        url: URL à tester
        features_expected: Liste des caractéristiques attendues
        model_loader: Instance de ModelLoader (optionnel)
    """
    logger.info(f"\n{'='*80}\nTest pour l'URL: {url}\n{'='*80}")

    # Créer un extracteur de caractéristiques
    extractor = WebFeatureExtractor(required_features=features_expected)

    # Mesurer le temps d'extraction
    start_time = time.time()
    features = extractor.extract_features(url)
    extraction_time = time.time() - start_time

    # Afficher les résultats
    logger.info(f"Temps d'extraction: {extraction_time:.2f} secondes")
    logger.info(f"Caractéristiques extraites: {len(features) - 1}")

    # Vérifier les caractéristiques manquantes
    missing = set(features_expected) - set(features.keys())
    if missing:
        logger.warning(f"Caractéristiques manquantes: {missing}")

        # Tester l'enrichissement
        enricher = FeatureEnrichment()
        start_time = time.time()
        enriched = enricher.enrich_features(url, features, list(missing))
        enrichment_time = time.time() - start_time

        logger.info(f"Temps d'enrichissement: {enrichment_time:.2f} secondes")
        logger.info(f"Caractéristiques enrichies: {len(enriched)}")

        # Ajouter les caractéristiques enrichies
        features.update(enriched)

        # Vérifier à nouveau les caractéristiques manquantes
        still_missing = set(features_expected) - set(features.keys())
        if still_missing:
            logger.error(
                f"Caractéristiques toujours manquantes après enrichissement: {still_missing}"
            )

    # Si un ModelLoader a été fourni, tester la prédiction
    if model_loader:
        # Créer un DataFrame avec les caractéristiques
        features_df = pd.DataFrame([features])

        # S'assurer que les features sont cohérentes
        start_time = time.time()
        processed_df = model_loader.ensure_feature_consistency(features_df)
        consistency_time = time.time() - start_time

        logger.info(
            f"Temps de validation de cohérence: {consistency_time:.2f} secondes"
        )

        # Effectuer la prédiction
        start_time = time.time()
        probas, predictions = model_loader.predict(processed_df)
        prediction_time = time.time() - start_time

        # Afficher les résultats
        logger.info(f"Temps de prédiction: {prediction_time:.2f} secondes")
        logger.info(
            f"Prédiction: {'Phishing' if predictions[0] == 1 else 'Légitime'} "
            f"(Probabilité de phishing: {probas[0]:.2%})"
        )

    # Afficher un résumé de toutes les caractéristiques
    logger.info("\nRésumé des caractéristiques:")
    for key, value in sorted(features.items()):
        if key != "url":
            logger.info(f"  {key}: {value}")

    return features


def main():
    """Fonction principale de test"""
    logger.info("Début des tests d'extraction de caractéristiques")

    # Liste des URLs à tester
    test_urls = [
        "https://www.google.com",  # Site légitime bien connu
        "http://example.com",  # Site HTTP simple
        "https://subdomain.example.co.uk",  # Sous-domaine avec TLD composite
        "http://malicious-site-with-very-long-domain-name-to-confuse-users.com",  # Nom de domaine long
        "https://paypa1.com/secure/login",  # Typosquatting classique
        "http://192.168.1.1",  # Adresse IP (souvent suspecte)
        "https://bit.ly/3rGH7q",  # URL raccourcie
        "https://secure-banking.com/%63%6C%69%65%6E%74",  # URL avec caractères obfusqués
        "non-existent-domain.xyz",  # Domaine probablement inexistant
        "https://website.com/login.php?redirectTo=https://malicious.com",  # URL avec redirection
    ]

    # Charger le modèle (si disponible)
    model_loader = None
    model_path = os.path.join(
        PROJECT_ROOT, "data", "processed", "lightgbm_phishing_model.pkl"
    )

    if os.path.exists(model_path):
        logger.info(f"Chargement du modèle depuis: {model_path}")
        model_loader = ModelLoader(model_path)
        if not model_loader.load():
            logger.error("Échec du chargement du modèle")
            model_loader = None
    else:
        logger.warning(f"Fichier modèle non trouvé: {model_path}")

    # Liste des caractéristiques attendues
    expected_features = [
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

    # Exécuter les tests pour chaque URL
    results = {}
    for url in test_urls:
        try:
            features = test_url(url, expected_features, model_loader)
            results[url] = {
                "success": True,
                "features_count": len(features) - 1,  # Exclure 'url'
                "expected_count": len(expected_features),
            }
        except Exception as e:
            logger.error(f"Erreur lors du test de {url}: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            results[url] = {"success": False, "error": str(e)}

    # Afficher le résumé des tests
    logger.info("\n\n" + "=" * 80)
    logger.info("RÉSUMÉ DES TESTS")
    logger.info("=" * 80)

    success_count = sum(1 for r in results.values() if r["success"])
    logger.info(
        f"Tests réussis: {success_count}/{len(test_urls)} ({success_count/len(test_urls):.1%})"
    )

    for url, result in results.items():
        status = "✓ RÉUSSI" if result["success"] else "✗ ÉCHEC"
        details = (
            f"{result.get('features_count', 0)}/{result.get('expected_count', 0)} features"
            if result["success"]
            else f"Erreur: {result.get('error', 'Inconnue')}"
        )
        logger.info(f"{status} - {url} - {details}")

    logger.info("\nFin des tests d'extraction de caractéristiques")


if __name__ == "__main__":
    main()
