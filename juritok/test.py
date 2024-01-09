import argparse
import sentencepiece as spm
import pandas as pd

def load_model(model_path):
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp

def print_characteristics(model):
    print("Vocab size: ", model.GetPieceSize())

def test_model(model, test_sentence):
    encoded = model.EncodeAsPieces(test_sentence)
    print("Sentence encoded :")
    print(model.EncodeAsPieces(test_sentence))
    print("Sentence encoded (IDs) :")
    print(model.EncodeAsIds(test_sentence))
    print("Sentence decoded :")
    print(model.DecodePieces(encoded))

def main():
    parser = argparse.ArgumentParser("training")
    parser.add_argument('--full_model', type=str, nargs='?', default="./law.model")
    parser.add_argument('--law_model', type=str, nargs='?', default="./full.model")
    parser.add_argument('--test_sentence', type=str, nargs='?', default="Le texte a été adopté le 3 février 2023.")
    args = parser.parse_args()

    print("Loading models")
    full_model = load_model(args.full_model)
    law_model = load_model(args.law_model)

    print("Model characteristics full model")
    print_characteristics(full_model)

    print("Model characteristics law model")
    print_characteristics(law_model)

    print("Result on test sentence")
    print("FULL MODEL")
    test_model(full_model, args.test_sentence)
    print("LAW MODEL")
    test_model(law_model, args.test_sentence)

if __name__ == '__main__':
    main()