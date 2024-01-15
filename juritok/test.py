import argparse
import sentencepiece as spm
import pandas as pd
from collections import Counter

def load_model(model_path):
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp

def print_characteristics(model):
    print("Vocab size: ", model.GetPieceSize())

def test_model_sentence(model, test_sentence):
    encoded = model.EncodeAsPieces(test_sentence)
    return encoded

def test_model_text(model, test_path):
    with open(test_path, 'r', encoding='utf-8') as f:
        test_sentences = f.readlines()
    return [test_model_sentence(model, sentence.strip()) for sentence in test_sentences]

def analysis(model, test_path):
    results = test_model_text(model, test_path)
    
    # Printing some examples of encoded sentences from test data
    print(f"Examples of encodings:")
    print(results[0], results[1], results[2])

    # Calculating vocabulary usage rate, i.e. number of distinct tokens used in test data compared to the vocab size 
    tokens = [token for sentence in results for token in sentence]
    vocab_usage_rate = len(set(tokens))/model.GetPieceSize()
    print(f"Vocab usage rate on test data: {vocab_usage_rate}")

    # Printing most frequent tokens (of at least 5 characters)
    long_tokens = [token for token in tokens if len(token)>=5]
    long_tokens_c = Counter(long_tokens)
    #tokens_c = Counter(tokens)
    print("Most frequent tokens (of at least 5 characters):")
    print(long_tokens_c.most_common(10))

def main():
    parser = argparse.ArgumentParser("training")
    parser.add_argument('--full_model', type=str, nargs='?', default="./full.model")
    parser.add_argument('--law_model', type=str, nargs='?', default="./law.model")
    parser.add_argument('--test_file_full', type=str, nargs='?', default="../../juritok_data/data_all_test.txt")
    parser.add_argument('--test_file_law', type=str, nargs='?', default="../../juritok_data/data_law_test.txt")
    args = parser.parse_args()

    print("Loading models...")
    full_model = load_model(args.full_model)
    law_model = load_model(args.law_model)

    print("Model characteristics full model")
    print_characteristics(full_model)

    print("Model characteristics law model")
    print_characteristics(law_model)

    print("Analysis of tokenized test data:")
    print("FULL MODEL")
    analysis(full_model, args.test_file_full)
    print("LAW MODEL")
    analysis(law_model, args.test_file_law)
    
if __name__ == '__main__':
    main()