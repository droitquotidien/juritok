import sentencepiece as spm
import csv

# Define the path to CSV file
csv_file_path = 'C:/Users/User/Desktop/jorf_2023.csv'

# Initialize an empty list to store the cleaned text data
texts = []

# Open and read the CSV file
with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
    csv_reader = csv.DictReader(csvfile, delimiter='|')

    for row in csv_reader:
        if len(row) > 0:
            last_column_key = list(row.keys())[-1]
            last_column_value = row[last_column_key].strip()
            texts.append(last_column_value)
text_to_write = "\n".join(texts)
with open("C:/Users/User/Desktop/train_data.txt", "w", encoding='utf-8') as f:
    f.write(text_to_write)

# Train SentencePiece tokenizer
spm.SentencePieceTrainer.Train('--input=C:/Users/User/Desktop/train_data.txt --model_prefix=m --vocab_size=8000')

# Load the trained SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load("m.model")

# Tokenize the texts
tokenized_texts = sp.EncodeAsPieces(texts)

# Print tokenized text for the first document
print(tokenized_texts[0])

