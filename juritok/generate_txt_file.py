import argparse
import pandas as pd
import sentencepiece as spm

def main():
    parser = argparse.ArgumentParser("training")
    parser.add_argument('--csv_path', type=str, nargs='?', default="../data/jorf_2023.csv")
    parser.add_argument('--txt_path', type=str, nargs='?', default="../data/jorf_2023.txt")
    args = parser.parse_args()
    df = pd.read_csv(args.csv_path,sep='|',usecols=[5],names=['text'])
    df = df.drop(df[df.text.apply(lambda string:'/' in str(string))].index)
    df.to_csv(args.txt_path, index=False, header=False)

if __name__ == '__main__':
    main()