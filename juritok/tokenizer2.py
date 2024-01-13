import sentencepiece as spm
import pandas as pd
import re

table = pd.read_csv('/mnt/c/Users/sacha/Downloads/jorf_2023.csv/jorf_2023.csv',sep="|")
data = table.iloc[:,5]

texte_complet = ""
for ligne in data :
    texte_complet += ligne

droit = re.findall(r'«([^»]*)»', texte_complet)

# Écrire le texte dans un fichier texte
with open('my_text_droit.txt', 'w', encoding='utf-8') as file:
    for ligne in droit :
        file.write(ligne + '\n')


#Train
spm.SentencePieceTrainer.train('--input=my_text_droit.txt --model_prefix=my_model2 --vocab_size=5000')

#Charger le modèle
sp = spm.SentencePieceProcessor()
sp.load('my_model2.model')

with open('my_text_droit.txt', 'r', encoding='utf-8') as file:
    # Utiliser read() pour lire tout le contenu du fichier
    texte_complet_droit = file.read()

# Tokeniser le texte complet
tokens = sp.encode_as_pieces(texte_complet_droit)

# Afficher les tokens
print(len(tokens))
print(tokens[:10])



