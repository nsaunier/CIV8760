# Installation des outils pour exécuter des programmes Python

Python est un langage de programmation puissant et facile à apprendre. Il dispose de structures de données de haut niveau et permet une approche simple mais efficace de la programmation orientée objet ([tutoriel officiel Python](https://docs.python.org/fr/3/tutorial/)).

Anaconda est la distribution recommandée pour installer l'interpréteur Python, les bibliothèques et les outils utiles pour écrire et exécuter des programmes Python. Si Anaconda ne fonctionne pas sur votre ordinateur, la solution la plus simple est d'utiliser les notebooks en ligne fournis sur la plateforme Colab de Google. 

## Anaconda
**Anaconda** est une distribution libre et open source des langages de programmation Python et R. Elle inclut un système de gestion de paquets **conda**. Conda crée, enregistre, charge et permet de changer facilement les environnements sur votre ordinateur local. 

![Anaconda](https://miro.medium.com/max/1600/1*_ozdDAB9qda1VuNGmw3CDA.png)

## Installer Anaconda

Télécharger selon le système d'opération : https://www.anaconda.com/products/individual

**Windows**
1. Double-cliquer le document .exe
2. Suivre les instructions à l'écran
3. Accepter les options par défaut

**MacOS**
1. Double-cliquer le document .pkg
2. Suivre les instructions à l'écran

**Linux**
1. Exécuter l'installateur .sh
2. Suivre les instructions du terminal

Il n'est pas nécessaire d'installer d'autres logiciels si vous avez réussi à installer Anaconda. Vous pourrez ajouter d'autres bibliothèques et outils avec Anaconda s'ils n'ont pas été installés initialement. 

## Python (inclus avec Anaconda)
Si on désire une installation minimale, on peut se contenter d'installer l'interpréteur et les bibliothèques standard du site officiel. 

**Installer Python**
1. Télécharger la version 3.8 de Python sur : https://www.python.org/downloads/
2. Suivre les instructions d'installation selon le système d'opération.

> [Lancer l'interpréteur Python](https://docs.python.org/fr/3/tutorial/interpreter.html#invoking-the-interpreter)

## IPython (inclus ou installable avec Anaconda)
Une des caractéristiques de Python est son interpréteur interactif. Il permet d'exécuter du code ligne par ligne pour tester très rapidement les idées sans avoir à créer des fichiers de code comme c'est le cas avec la plupart des langages de programmation. Cependant, l'interpréteur fourni avec la distribution standard de Python est limité pour une utilisation interactive étendue.

IPython est un projet visant à créer un environnement complet pour l'informatique interactive et exploratoire. Il comprend de nombreux sous-projets et a donné naissance, entre autres, aux notebooks Jupyter. La partie qui nous intéresse et que vous êtes encouragés à utiliser est l'interpréteur interactif IPython dans un terminal ou dans sa version QtConsole distribuée avec Anaconda ou Jupyter. 

IPython contient aussi:
* Un noyau pour Jupyter,
* Une architecture pour le calcul parallèle interactif fait désormais partie du package ipyparallel.

Pour en savoir plus:
> [Documentation IPython](https://ipython.readthedocs.io/en/stable/overview.html)
> [Introduction à IPython (EN)](https://ipython.readthedocs.io/en/stable/interactive/tutorial.html)

## Jupyter Notebook (inclus ou installable avec Anaconda)
Jupyter est un projet qui offre entre autres une application Internet qui permet de créer des notebooks. Un notebook se présente sous forme d'une page éditable dans un navigateur Internet. Chaque notebook se compose de sections (ou cellules) de texte (au format markdown) et de code Python. Il est sauvé dans un fichier avec l'extension `.ipynb`.

**Installer Jupyter (avec Anaconda)** 
1. Ouvrir Anaconda Navigator.
2. Cliquer sur Launch en dessous de Jupyter.
3. Suivre les instructions d'installation.

![Jupyter](https://miro.medium.com/max/4000/1*CrzFvh-ha0mkhUrA3q786A.png)

Pour en savoir plus :
> [Tutoriel sur Jupyter Notebook](http://python.lecoinduprogrammeur.org/2017/05/07/jupyter-notebook-ecrivez-executer-documentez-et-publiez-votre-code-python/)

## Colaboratory
Colaboratory, souvent raccourci en "Colab", est une version des notebooks Jupyter adaptée par Google, 
- sans aucune configuration ou installation sur votre ordinateur,
- avec un accès gratuit à des cartes graphiques,
- permettant un partage facile. 

Les fichiers de colab sont des fichiers Jupyter notebook `.ipynb` standard stockés dans l'espace disque Google Drive. 

![Colab](https://www.tutorialspoint.com/google_colab/images/new_notebook.jpg)

Pour en savoir plus:
> [Tutoriel sur Colaboratory](tutoriel-colab.md)

## Gestion des paquets (PIP, conda)

À venir
