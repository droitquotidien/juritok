import sentencepiece as sp
import os
import csv

# Write relevant data into a single txt file for training
txt_name = "data/data.txt"
if not os.path.exists(txt_name):
    with open(txt_name, 'w', encoding="utf-8") as txt_file:
        for year in range(2019, 2024):
            file_name = f"data/jorf_{year}.csv"
            with open(file_name, encoding="utf-8") as csv_file:
                reader = csv.reader(csv_file, delimiter='|')
                for row in reader:
                    text = row[-1]
                    if text[:3] != "fr/" and text[:6] != "<TABLE>":
                        txt_file.write(text + "\n")

# Initialize SentencePiece
sp.SentencePieceTrainer.train('--input=data/data.txt --model_prefix=m --vocab_size=1000')
sp_proc = sp.SentencePieceProcessor()
sp_proc.load('m.model')

# Show some tokens
vocabs = [sp_proc.id_to_piece(id) for id in range(sp_proc.get_piece_size())]
print(vocabs[:1000:100])

# Encoding / decoding demo
demo_text = 'En application du...'

encoded_pieces = sp_proc.encode_as_pieces(demo_text)
encoded_ids = sp_proc.encode_as_ids(demo_text)
print(encoded_pieces)
print(encoded_ids)

decoded_pieces = sp_proc.decode_pieces(encoded_pieces)
decoded_ids = sp_proc.decode_ids(encoded_ids)
print(decoded_pieces)
print(decoded_ids)