import argparse
import sentencepiece as spm
import pandas as pd

def main():
    parser = argparse.ArgumentParser("data_preprocessing")
    parser.add_argument('--csv_file', help="CSV file", type=str, nargs='?', default="../../juritok_data/jorf_2023.csv")
    args = parser.parse_args()

    print("Reading input CSV file")
    data = pd.read_csv(args.csv_file, sep = "|").iloc[:,5]
    data_all = "\n".join(data)
    data_law = "\n".join(data[data.str.startswith("«")])

    print("Writing output TXT files")
    with open(args.csv_file[:-4]+"_all.txt", "w", encoding="utf-8") as f:
        f.write(data_all)

    with open(args.csv_file[:-4]+"_law.txt", "w", encoding="utf-8") as f:
        f.write(data_law)

if __name__ == '__main__':
    main()