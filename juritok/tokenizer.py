import sentencepiece as sp
import pandas as pd

# On convertit les données en format txt
table = pd.read_csv('data/jorf_2023.csv',sep='|')
table.iloc[:, 5].to_csv('data/data.txt', index=False, header=False, sep='\t')

# On entraîne un modèle SentencePiece
sp.SentencePieceTrainer.train('--input=data/data.txt --model_prefix=m --vocab_size=1000')
sp_proc = sp.SentencePieceProcessor()
sp_proc.load('m.model')

vocabs = [sp_proc.id_to_piece(id) for id in range(sp_proc.get_piece_size())]
print(vocabs[:10])

# Exemple d'encodage et décodage
txt_sample = "Arrêté du 20 décembre 2023 modifiant l'arrêté du 30 janvier 2020 modifié relatif aux permis d'accès pour l'exercice de la pêche professionnelle dans le secteur de la baie de Granville"

encoded_pieces = sp_proc.encode_as_pieces(txt_sample)
encoded_ids = sp_proc.encode_as_ids(txt_sample)
print("encoded pieces:", encoded_pieces)
print("encoded ids:", encoded_ids)

decoded_pieces = sp_proc.decode_pieces(encoded_pieces)
decoded_ids = sp_proc.decode_ids(encoded_ids)
print("decoded pieces:", decoded_pieces)
print("decoded ids:", decoded_ids)
