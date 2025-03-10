# Analyse des données ESR — Application d'aide au pilotage

## Présentation

Cette application Streamlit fournit un outil complet d'analyse statistique et de visualisation de données pour l'enseignement supérieur et la recherche (ESR). Elle s'intègre au portail d'aide à la décision et permet l'exploration interactive des données, la création de visualisations avancées et la génération d'indicateurs personnalisés.

![Badge République Française](https://img.shields.io/badge/R%C3%A9publique-Fran%C3%A7aise-blue)

## Fonctionnalités principales

### 1. Analyse univariée
- **Variables qualitatives** : Tableaux de distribution, graphiques à barres horizontaux modernes, treemaps, dot plots, diagrammes radar, et lollipop plots
- **Variables quantitatives** : Statistiques descriptives, histogrammes, density plots, box plots et violin plots
- **Dashboards récapitulatifs** automatiques pour un aperçu rapide des caractéristiques principales des variables

### 2. Analyse bivariée
- **Croisement qualitatif-qualitatif** : Tableaux croisés avec tests du Chi² et visualisations adaptées
- **Croisement qualitatif-quantitatif** : Statistiques par groupe, comparaisons de distribution, tests ANOVA
- **Croisement quantitatif-quantitatif** : Nuages de points, corrélations (Pearson/Spearman), régressions linéaires

### 3. Analyse de séries temporelles
- Visualisation interactive des évolutions temporelles
- Agrégation par différentes fréquences (jour, semaine, mois, trimestre, année)
- Analyses de tendances et d'autocorrélations
- Comparaisons multi-séries avec groupement personnalisable

### 4. Exploration et filtrage des données
- Interface intuitive de filtrage des données dans la barre latérale
- Détection automatique des types de variables pour des analyses adaptées
- Analyse des valeurs manquantes et détection d'anomalies
- Aperçu paginé des données avec sélection des colonnes

### 5. Export de résultats
- Export des visualisations en haute résolution
- Génération de rapports automatisés au format HTML
- Export des données filtrées aux formats Excel, CSV et JSON
- Création d'indicateurs personnalisés avec stockage dans Grist

### 6. Interface utilisateur avancée
- Navigation par onglets pour une meilleure organisation
- Sélecteur de variables amélioré avec filtrage et prévisualisation
- Dashboards récapitulatifs pour une compréhension rapide des données
- Design conforme au Système Graphique de l'État français

## Technologies

Cette application utilise les technologies et bibliothèques suivantes :

- **Streamlit** : Framework pour l'interface utilisateur interactive
- **Plotly** : Visualisations interactives avancées
- **Pandas & NumPy** : Manipulation et analyse de données
- **SciPy & StatsModels** : Tests statistiques et analyses avancées
- **Scikit-learn** : Apprentissage automatique pour la détection d'anomalies et l'analyse prédictive
- **Grist API** : Stockage sécurisé des données et des indicateurs
- **Kaleido** : Export haute résolution des visualisations

## Intégration avec le portail d'aide à la décision

Cette application constitue le module d'analyse statistique du portail d'aide à la décision pour l'enseignement supérieur. Elle s'articule avec :

- **Le catalogue d'indicateurs** : Enrichissement avec de nouveaux indicateurs créés via l'analyse statistique
- **Les fiches d'analyse prospective** : Validation empirique des hypothèses et tendances identifiées
- **Les tableaux de bord par programme** : Exploration approfondie des données sous-jacentes
- **Les tableaux de bord par projet structurel** : Création d'indicateurs spécifiques pour mesurer l'impact des transformations

## Fonctionnalités d'intelligence artificielle

L'application intègre plusieurs fonctionnalités d'intelligence artificielle et d'apprentissage automatique :

- **Détection d'anomalies** : Utilisation d'algorithmes statistiques (Z-score, IQR) et de méthodes d'apprentissage non supervisé pour identifier les valeurs aberrantes
- **Classification automatique** : Détection des types de variables et suggestion d'analyses appropriées
- **Analyse prédictive** : Modélisation de tendances pour les séries temporelles
- **Optimisation des visualisations** : Choix automatique des formats de visualisation selon les caractéristiques des données

## Installation et déploiement

### Prérequis

- Python 3.8 ou supérieur
- Accès à une instance Grist (pour le stockage des données)

### Installation locale

1. Clonez ce dépôt :
   ```
   git clone https://github.com/votre-organisation/analyse-esr.git
   cd analyse-esr
   ```

2. Créez un environnement virtuel et installez les dépendances :
   ```
   python -m venv venv
   source venv/bin/activate  # Sur Windows : venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configurez les variables d'environnement pour l'API Grist :
   ```
   # Créez un fichier .streamlit/secrets.toml avec :
   grist_key = "votre_clé_api"
   grist_doc_id = "votre_id_de_document"
   ```

4. Lancez l'application :
   ```
   streamlit run streamlit_app.py
   ```

### Déploiement sur Streamlit Cloud

1. Créez un dépôt GitHub avec votre code et le fichier `requirements.txt`
2. Connectez-vous à [Streamlit Cloud](https://streamlit.io/cloud)
3. Déployez une nouvelle application en pointant vers votre dépôt
4. Configurez les secrets dans l'interface web de Streamlit Cloud

## Utilisation

### Analyse univariée

1. Sélectionnez l'onglet "Analyse univariée"
2. Choisissez une variable à analyser
3. Explorez le dashboard récapitulatif généré automatiquement
4. Sélectionnez le type de visualisation souhaité et personnalisez les options
5. Exportez les résultats ou créez un indicateur basé sur l'analyse

### Analyse bivariée

1. Sélectionnez l'onglet "Analyse bivariée"
2. Choisissez les variables X et Y à analyser
3. Explorez les statistiques et visualisations adaptées au type de variables
4. Interprétez les tests statistiques générés automatiquement
5. Exportez les résultats ou créez un indicateur basé sur l'analyse

### Séries temporelles

1. Sélectionnez l'onglet "Séries temporelles"
2. Choisissez la variable de temps et la variable de valeur
3. Optionnellement, sélectionnez une variable de groupement
4. Définissez la fréquence d'agrégation et la méthode
5. Analysez les tendances et exportez les résultats

## Contributeurs

- Équipe d'aide au pilotage de l'enseignement supérieur

## Licence

© Ministère de l'Enseignement Supérieur et de la Recherche - Tous droits réservés
