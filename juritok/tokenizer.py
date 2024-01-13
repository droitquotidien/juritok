import sentencepiece as spm
import pandas as pd

#table = pd.read_csv('/mnt/c/Users/sacha/Downloads/jorf_2023.csv/jorf_2023.csv',sep="|")
#data = table.iloc[:,5]

# Écrire le texte dans un fichier texte
#with open('my_text.txt', 'w', encoding='utf-8') as file:
#    for ligne in data:
#        file.write(ligne + '\n')


#Train
spm.SentencePieceTrainer.train('--input=my_text.txt --model_prefix=my_model --vocab_size=5000')

#Charger le modèle
sp = spm.SentencePieceProcessor()
sp.load('my_model.model')

with open('my_text.txt', 'r', encoding='utf-8') as file:
    # Utiliser read() pour lire tout le contenu du fichier
    texte_complet = file.read()

# Tokeniser le texte complet
tokens = sp.encode_as_pieces(texte_complet)

# Afficher les tokens
print(len(tokens))
print(tokens[:10])



