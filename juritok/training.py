import argparse
import sentencepiece as spm
import pandas as pd

def create_model(data, vocab_size, input_sentence_size, model_prefix, model_type = "bpe"):
    spm.SentencePieceTrainer.Train(input = data, vocab_size = vocab_size,
                                   input_sentence_size = input_sentence_size,
                                   model_prefix = model_prefix, model_type = model_type)

def main():
    parser = argparse.ArgumentParser("training")
    parser.add_argument('--full_data_file', help="TXT file", type=str, nargs='?', default="../../juritok_data/jorf_2023_all.txt")
    parser.add_argument('--law_data_file', help="TXT file", type=str, nargs='?', default="../../juritok_data/jorf_2023_law.txt")
    parser.add_argument('--vocab_size', type=int, nargs='?', default=1000)
    parser.add_argument('--input_sentence_size', type=int, nargs='?', default=100000)
    args = parser.parse_args()

    print("Creating models")
    create_model(args.full_data_file, args.vocab_size, args.input_sentence_size, "full")
    create_model(args.law_data_file, args.vocab_size, args.input_sentence_size, "law")

if __name__ == '__main__':
    main()