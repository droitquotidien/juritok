import pandas as pd
import sentencepiece as spm
from pathlib import Path

# Pre-processing: aggregate all data in a text file
raw_text_path = Path("data/raw_text.txt")
if not raw_text_path.exists():
    df = pd.read_csv("data/jorf_2023.csv", sep="|", header=None, skiprows=1)
    with open(raw_text_path, "w") as f:
        for i in range(len(df)):
            line = df.iloc[i][5]
            if line[:3] != "fr/" and line[:6] != "<TABLE>" and line[:6] != "ANNEXE": # Trim some not so relevant data
                f.write(str(df.iloc[i][5]) + "\n")

def show_infos(sp: spm.SentencePieceProcessor):
    """
    Print some relevant information about a model.
    """
    # Show most frequent token
    tokens = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]
    print(tokens[:50]) # Print the first 50 tokens

    test_texts = ["L'article 2 de la Constitution",
                  "Le ministre de l'intérieur et des outre-mer et le ministre délégué auprès du ministre de l'intérieur et des outre-mer"]
    for text in test_texts:
        print(text)
        encoded_tokens= sp.encode_as_pieces(text)
        encoded_ids = sp.encode_as_ids(text)
        print(encoded_tokens)
        print(encoded_ids)
        print("Decoded tokens: " + sp.decode_pieces(encoded_tokens))
        print("Decoded ids: " +sp.decode_ids(encoded_ids))
    print("\n")

# Train a model with 2000 tokens
if not Path("large_model.model").exists():
    spm.SentencePieceTrainer.train(f"--input={raw_text_path} --model_prefix=large_model --vocab_size=2000")
large_sp = spm.SentencePieceProcessor()
large_sp.load('large_model.model')
print("Large model with 2000 tokens\n")
show_infos(large_sp)


# Train a model with only 100 tokens
if not Path("small_model.model").exists():
    spm.SentencePieceTrainer.train(f"--input={raw_text_path} --model_prefix=small_model --vocab_size=100")
small_sp = spm.SentencePieceProcessor()
small_sp.load('small_model.model')
print("Small model with 100 tokens")
show_infos(small_sp)
