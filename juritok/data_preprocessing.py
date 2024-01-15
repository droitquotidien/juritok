import argparse
import os
import pandas as pd
from collections import Counter

def vocab_size_analysis(training_data, vocab_coverage):
    training_data_list = training_data.split()
    dict_occurences = Counter(training_data_list)
    total_num_words = len(training_data_list)
    total_num_distinct_words = len(dict_occurences.keys())
    print(f"Total number of words: {total_num_words}")
    print(f"Total number of distinct words: {total_num_distinct_words}")
    
    actual_coverage = 0
    vocab_size = 0
    for _, occ in dict_occurences.most_common():
        actual_coverage += occ / total_num_words
        vocab_size += 1
        if actual_coverage >= vocab_coverage:
            break
    print(f"Vocab size to get a {vocab_coverage} coverage: {vocab_size}")

def main():
    parser = argparse.ArgumentParser("data_preprocessing")
    parser.add_argument('--data_folder', help="Folder containing all (and only) CSV files", type=str, nargs='?', default="../../juritok_data/")
    parser.add_argument('--vocab_coverage', help="Proportion of occurences that we want to cover", type=float, nargs='?', default=0.95)
    args = parser.parse_args()

    print("Reading files in input folder...")
    files = [args.data_folder + f for f in os.listdir(args.data_folder) if os.path.isfile(args.data_folder + f)]
    data_all_train = ""
    data_law_train = ""

    # TRAINING DATA
    for f in files[:-1]: # All but the last file are used for training
        data = pd.read_csv(f, sep = "|", dtype=str).iloc[:,5].astype(str)
        data_all_train = data_all_train + "\n" + "\n".join(data)
        data_law_train = data_law_train + "\n" + "\n".join(data[data.str.startswith("«")])

    # TEST DATA
    data_test = pd.read_csv(files[-1], sep = "|", dtype=str).iloc[:,5].astype(str) # The last CSV file is used for test
    data_all_test = "\n".join(data_test)
    data_law_test = "\n".join(data_test[data_test.str.startswith("«")])

    print("Writing output TXT files...")
    # We do not perform additional pre-processing tasks since SentencePiece is capable of handling punctuation, special characters and is case-sensitive
    with open(args.data_folder+"data_all_train.txt", "w", encoding="utf-8") as f:
        f.write(data_all_train)

    with open(args.data_folder+"data_law_train.txt", "w", encoding="utf-8") as f:
        f.write(data_law_train)

    with open(args.data_folder+"data_all_test.txt", "w", encoding="utf-8") as f:
        f.write(data_all_test)

    with open(args.data_folder+"data_law_test.txt", "w", encoding="utf-8") as f:
        f.write(data_law_test)
   
    print("Determining appropriate vocab size from training data...")
    print("On ALL data (patch + law):")
    vocab_size_analysis(data_all_train, args.vocab_coverage)

    print("On LAW data only:")
    vocab_size_analysis(data_law_train, args.vocab_coverage)

if __name__ == '__main__':
    main()