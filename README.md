# juritok
Tokenisation des textes du JO et des textes consolidés

Petits tests pour essayer de comprendre comment fonctionne la tokenisation en se basant sur le fichier de données juridiques jorf_2023.csv

On commence par lire le fichier csv, en séparant les données selon le séparateur |, et on sait que les données de loi intéressantes seront dans la colonne 6.

Les données texte sont recopiées dans un fichier à part, puis en utilisant Sentencepiece, on entraîne un modèle sur la base du fichier de données entier. On teste ensuite le modèle en affichant les résultats d'une tokenisation des 10 dernières lignes de loi seulement.

On voit dans le fichier "sortie_du_code.txt" ce que le terminal renvoie une fois qu'on a exécuté le code, avec les 10 dernières lignes de loi, leur version tokenisée, et leur version reconstruite! Cela fonctionne.
