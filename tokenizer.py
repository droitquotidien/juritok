import pandas as pd
import sentencepiece as sp

filepath = r"C:\Users\xavde\OneDrive\Bureau\NLP\juritok\juritok\jorf_2018.csv"

data = pd.read_csv(filepath,  sep="|", header=None, dtype="str", encoding="utf-8")

data = data.iloc[:,5]

data[:406000].to_csv("train.txt", sep="\t", index=False, header=False)

data[406000:].to_csv("test.txt", sep="\t", index=False, header=False)

sp.SentencePieceTrainer.train('--input=train.txt --model_prefix=model --vocab_size=1000')


with open("test.txt", "r", encoding="utf-8") as file:
    test_text = file.read().replace("\xa0", " ").replace("\n", " ")

tokenizer = sp.SentencePieceProcessor()

tokenizer.load("model.model")

encoded = tokenizer.EncodeAsPieces(test_text)

print(tokenizer.decode_pieces(encoded))