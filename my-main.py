import pandas as pd
import sentencepiece as spm
import sys

path = "jorf_2023.csv"
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    print("Usage: python my-main.py [path_to_jorf]\n(Default jorf_2023.csv)")

df = pd.read_csv(
    path, sep="|", names=["ID2", "Unkown", "N° loi", "N° article", "Contenu"]
).reset_index(names=["ID1"])

with open("10000_lines.txt", "w", encoding="utf-8") as f:
    for i, l in enumerate(df["Contenu"]):
        if i >= 10000:
            break
        f.write(l)
        f.write("\n")

with open("all_lines.txt", "w", encoding="utf-8") as f:
    for i, l in enumerate(df["Contenu"]):
        f.write(l)
        f.write("\n")

print(
    f"Training tokenizers, with {len(df)} and then 10000 lines (takes 1min to run)..."
)
spm.SentencePieceTrainer.train(
    input="all_lines.txt", model_prefix="m", vocab_size=11002, minloglevel=4
)
spm.SentencePieceTrainer.train(
    input="10000_lines.txt",
    model_prefix="m_lite",
    vocab_size=int(11002 * 0.75),
    minloglevel=2,
)
print("Done\n\nComparing the two tokenizers (normal, then lite):")
sp = spm.SentencePieceProcessor()
sp_lite = spm.SentencePieceProcessor()
sp.load("m.model")
sp_lite.load("m_lite.model")
df_sample = df.sample(10)
for txt in df_sample["Contenu"]:
    print(txt)
    print(sp.encode_as_pieces(txt))
    print(sp_lite.encode_as_pieces(txt))
    print()
