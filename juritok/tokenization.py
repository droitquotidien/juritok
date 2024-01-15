import csv
import sentencepiece as spm

tableau = []

f = open('juritok/loi.txt', 'w', encoding='utf-8')
with open('juritok/jorf_2023.csv', 'r', newline='', encoding='utf-8') as fichier_csv:
    lecteur_csv = csv.reader(fichier_csv, delimiter='|')
    
    for ligne in lecteur_csv:
        tableau.append(ligne)
        f.write(ligne[5] + '\n')
        

spm.SentencePieceTrainer.train(input='juritok/loi.txt', model_prefix='spm', vocab_size=1000)

# Chargement du modèle SentencePiece
sp = spm.SentencePieceProcessor(model_file='spm.model')

# Tokenisation d'une phrase
phrase = "Ceci est un exemple de tokenisation avec SentencePiece."
tokens = sp.encode_as_pieces(phrase)

# Affichage des tokens
print("Phrase originale:", phrase)
print("Tokens:", tokens)

# Décodage des tokens
phrase_reconstruite = sp.decode_pieces(tokens)
print("Phrase reconstruite:", phrase_reconstruite)