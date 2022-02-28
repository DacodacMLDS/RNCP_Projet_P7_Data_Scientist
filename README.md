<h1>Projet 7: Implémentez un modèle de scoring</h1>
  
![My Image](pretadepenser.jpg)

<h2>Contexte et problématique du projet</h2>

Vous êtes Data Scientist au sein d'une société financière, nommée "Prêt à dépenser",  qui propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.

L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner.

Prêt à dépenser décide donc de développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement. 

<h2>But et intérêt du projet</h2>

Construction d'un modèle de scoring pour donner une prédiction sur la probabilité de faillite d'un client de façon automatique, à partir des données suivantes : https://www.kaggle.com/c/home-credit-default-risk/data

- Préparation des données à partir d'un kernel Kaggle existant en l'adaptant aux besoins du projet par feature engineering et feature selection.
- Elaboration d'un modèle de classification, optimisation des hyper-paramètres.
- Mise en oeuvre d'une métrique personnalisée sous la forme d'un fonction de revenu net.
- Choix du modèle de classification optimal suivant les métriques AUC et la fonction de revenu net.
- Ajustement du seuil de probabilité (threshold) par rapport à la métrique personnalisée.
- Analyse de l'importance des variables avec SHAP.
- Rédaction d'une note méthodologique expliquant la méthodologie d'entraînement du modèle, la métrique personnalisée, l'algorithme d'optimisation et la métrique d'évaluation, l’interprétabilité globale et locale du modèle, les limites et les améliorations possibles.

Construction d'un dashboard interactif (avec Streamlit) à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle, et d’améliorer la connaissance client des chargés de relation client:

- Visualisation du score et interprétation de ce score pour chaque client de façon intelligible pour une personne non experte en data science.
- Visualisation des informations descriptives relatives à un client (via un système de filtre).
- Comparaison des informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.

<h2>Compétences évaluées</h2>

- Présenter son travail de modélisation à l'oral
- Réaliser un dashboard pour présenter son travail de modélisation
- Rédiger une note méthodologique afin de communiquer sa démarche de modélisation
- Utiliser un logiciel de version de code pour assurer l’intégration du modèle
- Déployer un modèle via une API dans le Web

<h2>Contenu du dépôt GitHub</h2>

- README.md: fichier présentation projet

- pretadepenser.jpg : image illustration README.md

- p7_acces_heroku.txt: url pour l'accès à l'API de prédiction Flask (https://davidp7apiflask.herokuapp.com) et au dashboard interactif (https://davidp7dashboard.herokuapp.com)

- Répertoire "Notebooks": 
  - fichier "P7_01_Notebook_EDA_Feature_Engineering.ipynb": fichier notebook Jupyter en Python pour le traitement des données et le feature enginneering à partir d'un kernel Kaggle existant 
  - fichier "P7_02_Notebook-Modeling.ipynb": fichier notebook Jupyter en Python pour la modélisation
  - fichier "P7_03_Notebook-Preparation_Dashboard.ipynb": fichier notebook Jupyter en Python pour la préparation du dashboard interactif
  
- Répertoire "Deploiement_web":
  - P7_07_deploiement_heroku.txt : fichier décrivant la procédure de déploiement d'une API de prédiction Flask et d'un dashboard interactif Streamlit sur un serveur Heroku
  - Répertoire "API_prediction": liste des répertoires / fichiers pour le déploiment de l'API de prédiction Flask sur Heroku
  - Répertoire "Dashboard": liste des répertoires / fichiers pour le déploiment du dashboard interactif Streamlit sur Heroku  
  
- Répertoire "Soutenance":
  - fichier "P7_04_soutenance_projet_ppt.ppt": fichier support soutenance projet Powerpoint
  - fichier "p7_04_soutenance_projet_pdf.pdf": fichier support soutenance projet PDF
  - fichier "projet_p7_oc_ds.mp4": video soutenance projet P7 (mp4)
  - fichier "p7_05_note_methodologique.doc": fichier note méthodologique sur modélisation Word
  - fichier "Projet 7 valide - Implémentez un modèle de scoring - OC.pdf": preuve de validation du projet P7
