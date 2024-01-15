import argparse
import sentencepiece as spm
import pandas as pd

def create_model(data, vocab_size, model_prefix, model_type = "bpe"):
    spm.SentencePieceTrainer.Train(input = data, vocab_size = vocab_size,
                                   model_prefix = model_prefix, model_type = model_type)

def main():
    parser = argparse.ArgumentParser("training")
    parser.add_argument('--full_data_file', help="TXT file", type=str, nargs='?', default="../../juritok_data/data_all_train.txt")
    parser.add_argument('--law_data_file', help="TXT file", type=str, nargs='?', default="../../juritok_data/data_law_train.txt")
    parser.add_argument('--full_data_vocab_size', type=int, nargs='?', default=40000) #Default value determined with preprocessing script, requiring a 0.95 coverage
    parser.add_argument('--law_data_vocab_size', type=int, nargs='?', default=15000) #Default value determined with preprocessing script, requiring a 0.95 coverage
    args = parser.parse_args()

    print("Creating models")
    create_model(args.full_data_file, args.full_data_vocab_size, "full")
    create_model(args.law_data_file, args.law_data_vocab_size, "law")

if __name__ == '__main__':
    main()