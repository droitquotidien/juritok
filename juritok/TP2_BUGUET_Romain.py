import pandas as pd
import sentencepiece as spm

# Lecture des fichiers CSV et stockage dans des DataFrames Pandas
df1 = pd.read_csv('jorf_2019.csv', sep='|')
df2 = pd.read_csv('jorf_2020.csv', sep='|')
df3 = pd.read_csv('jorf_2021.csv', sep='|')
df4 = pd.read_csv('jorf_2022.csv', sep='|')
df5 = pd.read_csv('jorf_2023.csv', sep='|')

# Concaténation des DataFrames df1, df2, df3, df4 and df5 en un seul DataFrame df
df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

# Extraction de la dernière colonne (là où se trouve le texte)
text = df.iloc[:, -1]

# Conversion des valeurs du DataFrame 'text' en chaînes de caractères
text_str = text.astype(str)

text_file = 'temp.txt'
text.to_csv(text_file, header=False, index=False)

# Entraînement de SentencePiece à partir du fichier texte 'text_file'
spm.SentencePieceTrainer.train(input=text_file, model_prefix='m', vocab_size=10000)

# Charge le modèle SentencePiece préalablement entraîné
sp = spm.SentencePieceProcessor()
sp.load('m.model')

# Exemple de segmentation en subwords et d'encodage en identifiants numériques
print(sp.encode_as_pieces('Ceci est un test'))
print(sp.encode_as_ids('Ceci est un test'))