import pandas as pd
from pathlib import Path
import sentencepiece as spm
import argparse
import numpy as np


# https://github.com/google/sentencepiece/blob/master/doc/options.md
def build_trainer(sentence_array, vocab_size=1000, input_sentence_size=100000):
    input_text = "\n\n".join(sentence_array)

    print("Writing input text")
    with open("jorf_2023.txt", "w", encoding="utf-8") as f:
        f.write(input_text)

    print("Training SentencePiece")
    # spm.SentencePieceTrainer.Train('--input=jorf_2023.txt --model_prefix=juritok --vocab_size=32000 --model_type=bpe --character_coverage=1.0 --pad_id=0 --unk_id=1 --bos_id=-1 --eos_id=-1 --pad_piece=[PAD] --unk_piece=[UNK] --bos_piece=[BOS] --eos_piece=[EOS] --user_defined_symbols=[SEP],[CLS],[MASK]')
    spm.SentencePieceTrainer.Train(
        f"--input=jorf_2023.txt "
        f"--vocab_size={vocab_size} "
        f"--model_prefix=spm "
        f"--input_sentence_size={input_sentence_size} "
        f"--model_type=bpe "
    )

    print("Loading SentencePiece")
    sp = spm.SentencePieceProcessor()
    sp.Load("spm.model")

    return sp

def test_model(sp, sentence):
    encoded = sp.EncodeAsPieces(sentence)
    print(sp.EncodeAsPieces(sentence))
    print(sp.EncodeAsIds(sentence))
    print(sp.DecodePieces(encoded))

def get_data():
    path = Path(__file__) / "../jorf_2023.feather"
    path_csv = Path(__file__) / "../jorf_2023.csv"

    if not path.exists():
        print("Converting CSV to Feather")
        csv = pd.read_csv(path_csv, sep='|', encoding='utf-8', low_memory=False, header=None)
        csv.to_feather(path)

    print("Reading Feather")
    data = pd.read_feather(path)
    print(data.head())
    return data[5]

def keep_law_only(data):
    return data[data.str.startswith("«")]

def count_different_words(sentence):
    if isinstance(sentence, pd.Series):
        sentence = " ".join(sentence)
    words = set()
    words.update(sentence.split())
    return len(words)


def main():
    print("JURITOK - Main")

    argparser = argparse.ArgumentParser()
    args = argparser.parse_args()

    data = get_data()

    data = data.iloc[:1000]
    data_law = keep_law_only(data)

    sp_all = build_trainer(data)
    sp_law = build_trainer(data_law)

    test_sentence = "« Le présent décret entre en vigueur le 1er janvier 2023. »"

    print()
    print()

    # print("==== PARAMETERS ====")
    # print()
    # print()

    print("==== TESTING MODEL WITH ALL JORF ====")
    print("Number of different words: ", count_different_words(data))
    print("Vocab size: ", sp_all.GetPieceSize())
    test_model(sp_all, test_sentence)

    print()
    print()

    print("==== TESTING MODEL WITH LAW ARTICLES ONLY ====")
    print("Number of different words: ", count_different_words(data_law))
    print("Vocab size: ", sp_law.GetPieceSize())
    test_model(sp_law, test_sentence)


    

