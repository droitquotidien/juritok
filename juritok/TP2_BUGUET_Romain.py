import pandas as pd
import sentencepiece as spm

# Lecture des fichiers CSV et stockage dans des DataFrames Pandas 
df18 = pd.read_csv('jorf_2018.csv', sep='|')
df19 = pd.read_csv('jorf_2019.csv', sep='|')
df20 = pd.read_csv('jorf_2020.csv', sep='|')
df21 = pd.read_csv('jorf_2021.csv', sep='|')
df22 = pd.read_csv('jorf_2022.csv', sep='|')
df23 = pd.read_csv('jorf_2023.csv', sep='|')

# Concaténation des DataFrames df18, df19, df20, df21 et df22 en un seul DataFrame df
# Le DataFrame df23 sera utilisé pour effectuer des tests
df = pd.concat([df18, df19, df20, df21, df22], ignore_index=True)

# Extraction de la dernière colonne de df (là où se trouve le texte)
last_column = df.iloc[:, -1]

# Conversion des valeurs du DataFrame 'last_column' en chaînes de caractères
text = last_column.astype(str)

text_file = 'temp.txt'
text.to_csv(text_file, header=False, index=False)

# Entraînement de SentencePiece à partir du fichier texte 'text_file'
# avec un vocabulaire de 10000 tokens
spm.SentencePieceTrainer.train(input=text_file, model_prefix='m', vocab_size=10000)

# Charge le modèle SentencePiece préalablement entraîné
sp = spm.SentencePieceProcessor()
sp.load('m.model')

# Exemple de segmentation en subwords et d'encodage en identifiants numériques
extract = str(df23.iloc[:10,-1])
subwords = sp.encode_as_pieces(extract)
idnum = sp.encode_as_ids(extract)
print(subwords)
print(idnum)

# Exemple de décodage
print(sp.decode_pieces(subwords))
print(sp.decode_ids(idnum))
