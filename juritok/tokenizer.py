import argparse
import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser("training")
    parser.add_argument('--data', type=str, nargs='?', default="../data/jorf_2023.txt")
    parser.add_argument('--vocab_size', type=int, nargs='?', default=40000) #Default value determined with preprocessing script, requiring a 0.95 coverage
    args = parser.parse_args()

    print("Creating models")
    spm.SentencePieceTrainer.Train(
        input = args.data, vocab_size = args.vocab_size, model_prefix = "m", model_type = "bpe"
    )

if __name__ == '__main__':
    main()