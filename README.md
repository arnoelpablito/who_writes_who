# Qui écrit qui ?

## Mise en scène des personnages et genre de l’auteur

**Arno De Sousa-Kornhauser**
*Décembre 2025*

## Résumé

Ce projet étudie dans quelle mesure la mise en scène des personnages dans un roman permet de décrire et de prédire des différences entre écrivaines et écrivains.

À partir de fichiers `.book` produits par **BookNLP**, nous construisons des variables agrégées portant sur :

* la composition du *cast* ;
* les rôles syntaxiques des personnages ;
* certains champs lexicaux ;
* la position des personnages dans le texte.

Nous appliquons ensuite ces mesures à un corpus d’environ **2 920 romans français**.

Le projet présente d’abord plusieurs régularités descriptives, puis entraîne des modèles de classification supervisée visant à prédire le genre de l’auteur. Les résultats mettent en évidence des différences statistiques entre autrices et auteurs, mais également un fort recouvrement entre les deux groupes, qui invite à une interprétation prudente.

## Fichiers

* [`descriptive.ipynb`](descriptive.ipynb) contient le code utilisé pour l’analyse descriptive des données, correspondant aux **parties 2 et 3** de [`rapport.pdf`](rapport.pdf).

* [`predict.ipynb`](predict.ipynb) contient le code utilisé pour la prédiction du genre de l’auteur, en s’appuyant sur [`features.py`](features.py). Cette analyse correspond à la **partie 4** de [`rapport.pdf`](rapport.pdf).

* [`features.py`](features.py) contient les fonctions utilisées pour construire les variables mobilisées dans les modèles de prédiction.

## Données

Les données ne sont pas directement disponibles dans ce dépôt.

Pour y accéder, contactez : [arnodsk@gmail.com](mailto:arnodsk@gmail.com).
