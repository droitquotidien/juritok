import sentencepiece as spm
import csv
import pandas as pd
  
csv_file_path = 'jorf_2023.csv'

# Lecture du csv
df = pd.read_csv(csv_file_path, sep='|', encoding='utf-8')

#On met dans une string en gardant uniquement le texte (6eme colonne)
valeurs_6_colonne = df.iloc[:, 5].tolist()
training_text_string = ''.join(map(str, valeurs_6_colonne))

#On met le texte d'entrainement dans un .txt pour pouvoir le mettre en entrée de SentencePieceProcessor 
temp_text_file_path = 'temp.txt'
with open(temp_text_file_path, 'w', encoding='utf-8') as temp_file:
    temp_file.write(training_text_string)

# On entraine le modele sentencepiecetrainer
spm.SentencePieceTrainer.train(f'--input={temp_text_file_path} --model_prefix=m --vocab_size=500')
sp = spm.SentencePieceProcessor()
sp.load('m.model')

# test encode
sub_units = sp.encode_as_pieces("Les agents exerçant les fonctions de secrétaire général de mairie bénéficient d'un avantage spécifique d'ancienneté pour le calcul de l'ancienneté requise au titre de l'avancement d'échelon")
ids = sp.encode_as_ids("Les agents exerçant les fonctions de secrétaire général de mairie bénéficient d'un avantage spécifique d'ancienneté pour le calcul de l'ancienneté requise au titre de l'avancement d'échelon")
print(sub_units)
print(ids)

# test decode 
print(sp.decode_pieces(sub_units))
print(sp.decode_ids(ids))


