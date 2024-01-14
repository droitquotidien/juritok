from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
import pandas as pd 
from pathlib import Path

def csv_tokenizer_training(csv: str | Path ) -> None:
    df = pd.read_csv(csv, sep="|", header=None, dtype="str", encoding="utf-8")
    series = df.iloc[:, 5]
    series.to_csv("training_txt.txt", sep="\t", index=False, header=False)

    SentencePieceTrainer.Train("--input=training_txt.txt "
        "--vocab_size=1000 "
        "--model_prefix=spm "
        "--model_type=bpe ")

def csv_tokenizer_test(csv: str | Path ) -> [str, str]:
    df = pd.read_csv(csv, sep="|", header=None, dtype="str", encoding="utf-8")
    series = df.iloc[:, 5]
    text = ''
    for serie in series :
        text+= str(serie) +'/n'
    #text = "/n".join(series)
    sp = SentencePieceProcessor()
    sp.Load("spm.model")
    tokens = sp.EncodeAsPieces(text)
    rebuilt = sp.DecodePieces(tokens)
    return tokens, rebuilt

if __name__ == "__main__":
    csv_tokenizer_training(Path(__file__) / "../jorf_2023.csv")
    tokens, result = csv_tokenizer_test(Path(__file__) / "../jorf_2022.csv")
    print(tokens)
    print(result)