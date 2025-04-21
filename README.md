# Application de Détection de Phishing

Une application web robuste pour détecter les URLs de phishing à l'aide d'un modèle de machine learning LightGBM. 

## Aperçu

Cette application permet d'analyser des URLs pour déterminer si elles sont potentiellement malveillantes (phishing) ou légitimes. Elle utilise diverses caractéristiques extraites des URLs et des informations complémentaires obtenues via une API externe (DeepSeek) pour effectuer une prédiction précise.

## Fonctionnalités

- **Analyse d'URL en temps réel** : Entrez une URL pour obtenir une prédiction instantanée
- **Exploration des données** : Visualisez et comprenez le jeu de données utilisé pour l'entraînement
- **Prétraitement des données** : Découvrez les techniques de prétraitement appliquées au jeu de données
- **Performances du modèle** : Explorez les métriques d'évaluation et l'importance des caractéristiques
- **Interface intuitive** : Navigation facile et visualisations interactives

## Structure du projet

```
phishing_detection_app/
│
├── README.md                     # Documentation du projet
├── requirements.txt              # Dépendances du projet
├── main.py                       # Point d'entrée de l'application Streamlit
│
├── data/                         # Dossier pour les données
│   ├── raw/                      # Données brutes
│   │   └── Phishing_Legitimate_full.csv
│   └── processed/                # Données transformées
│       └── lightgbm_phishing_model.pkl
│
├── src/                          # Code source de l'application
│   ├── __init__.py
│   ├── config.py                 # Configuration de l'application
│   ├── preprocessing/            # Modules de prétraitement
│   │   ├── __init__.py
│   │   └── feature_extractor.py  # Extraction des caractéristiques d'URL
│   │
│   ├── model/                    # Modules liés au modèle
│   │   ├── __init__.py
│   │   └── model_loader.py       # Chargement du modèle sauvegardé
│   │
│   └── api/                      # Intégrations API
│       ├── __init__.py
│       └── deepseek_client.py    # Client pour l'API DeepSeek
│
└── pages/                        # Pages de l'application Streamlit
    ├── __init__.py
    ├── home.py                   # Page d'accueil
    ├── data_exploration.py       # Page d'exploration des données
    ├── preprocessing.py          # Page de prétraitement
    ├── model_performance.py      # Page de performance du modèle
    └── prediction.py             # Page de prédiction en temps réel
```

## Installation

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/votre-nom/phishing-detection-app.git
   cd phishing-detection-app
   ```

2. Créez un environnement virtuel Python et activez-le :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows : venv\Scripts\activate
   ```

3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

4. Créez les dossiers nécessaires :
   ```bash
   mkdir -p data/raw data/processed
   ```

5. Placez vos fichiers de données :
   - Copiez `Phishing_Legitimate_full.csv` dans `data/raw/`
   - Copiez `lightgbm_phishing_model.pkl` dans `data/processed/`

## Utilisation

Lancez l'application Streamlit :
```bash
streamlit run main.py
```

L'application sera accessible dans votre navigateur à l'adresse `http://localhost:8501`.

## Intégration avec DeepSeek API

L'application peut utiliser l'API DeepSeek pour obtenir des informations supplémentaires sur les URLs qui ne peuvent pas être extraites directement. Pour activer cette fonctionnalité :

1. Obtenez une clé API DeepSeek
2. Définissez la clé API comme variable d'environnement :
   ```bash
   export DEEPSEEK_API_KEY="votre_clé_api"
   # Sur Windows : set DEEPSEEK_API_KEY=votre_clé_api
   ```

Si aucune clé API n'est fournie, l'application fonctionnera en mode simulation, générant des données fictives pour la démonstration.

## Technologies utilisées

- **Python** : Langage de programmation principal
- **Streamlit** : Framework pour l'interface web
- **LightGBM** : Modèle de machine learning
- **Pandas & NumPy** : Manipulation et analyse des données
- **Plotly & Altair** : Visualisations interactives
- **scikit-learn** : Prétraitement et évaluation du modèle

## Fonctionnement du modèle

Le modèle LightGBM a été entraîné sur un ensemble de données contenant des URLs légitimes et de phishing. Le processus d'entraînement comprend plusieurs étapes :

1. **Prétraitement** : Transformation des distributions (PowerTransformer), standardisation, détection d'anomalies (IsolationForest)
2. **Équilibrage** : Technique SMOTE pour équilibrer les classes
3. **Sélection de caractéristiques** : RFECV pour identifier les caractéristiques les plus pertinentes
4. **Optimisation des hyperparamètres** : GridSearchCV pour trouver les meilleurs paramètres

Le modèle final atteint une précision supérieure à 94% sur l'ensemble de test.

## Contribuer

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

1. Fork du projet
2. Création d'une branche (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit des changements (`git commit -m 'Ajout d'une nouvelle fonctionnalité'`)
4. Push vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Ouverture d'une Pull Request


## Contact

Pour toute question ou suggestion, n'hésitez pas à me contacter.

---

Développé avec Nejmeddin Ben Maatoug , Informatic ingeneer.