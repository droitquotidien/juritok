import csv
import re
import sentencepiece as spm

# On recopie le fichier de données 2023 dans un fichier de résultats créé pour l'occasion
f = open('resultats.txt', 'w', encoding='utf-8')
fichier ='jorf_2023.csv'

with open(fichier, 'r', newline='', encoding='utf-8') as donnees_entree:
    donnees_separees = csv.reader(donnees_entree, delimiter='|') # on sépare en utilisant |
    for ligne in donnees_separees:
            f.write(ligne[5].lower()+"\n") # seule la 6e colonne nous intéresse

f.close()

# On entraîne les données
spm.SentencePieceTrainer.train(input='resultats.txt', model_prefix='jorf_2023', vocab_size=1000)

# On teste le modèle
sp = spm.SentencePieceProcessor(model_file='jorf_2023.model')

# On tokenise les 10 dernières lignes de notre fichier de données pour tester
dernieres_lignes = ""
with open(fichier, 'r', newline='', encoding='utf-8') as donnees_test:
    donnees_separees = csv.reader(donnees_test, delimiter='|') # on sépare en utilisant |
    k = 1
    for ligne in donnees_separees:
        if k<=10:
            dernieres_lignes += ligne[5]+"\n" # seule la 6e colonne nous intéresse
            k += 1

tokens = sp.encode_as_pieces(dernieres_lignes.lower())

# On affiche les tokens
print("Phrases originales:", dernieres_lignes)
print("Tokens:", tokens)

# On reconstruit les phrases à partir des tokens
phrase_reconstruite = sp.decode_pieces(tokens)
print("Phrase reconstruite:", phrase_reconstruite)