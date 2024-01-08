import sentencepiece as spm
import pandas as pd

file_path = "jorf_2023.csv"

data = pd.read_csv(file_path, sep="|")
last_column = data.iloc[:, -1]
combined_text = last_column.str.cat(sep=' ')

with open("train_data.txt", "w") as f:
    f.write(combined_text)

# train model
spm.SentencePieceTrainer.train(input="train_data.txt", model_prefix='m', vocab_size=500, model_type='bpe')

# load model
sp = spm.SentencePieceProcessor()
sp.load('m.model')

# test model
sample_text = "1° A la première phrase du second alinéa de l'article 196 B, le montant : « 6 042 € » est remplacé par le montant : « 6 368 € » ;"
encoded_text = sp.encode_as_pieces(sample_text)
print("Encoded Text:", encoded_text)
