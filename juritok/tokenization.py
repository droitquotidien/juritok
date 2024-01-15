import sys
import pandas as pd
import sentencepiece as spm

if len(sys.argv) != 2:
	print(f"You must provide the path to the .csv files. You can run this script like:\nPython3 ./juritok/tokenization.py ./data")
	sys.exit()

PATH = sys.argv[1]

def flatten(xss):
    return [x for xs in xss for x in xs]

def read_csv_and_extract_text(file_name):
	try:
		print(f"Reading {file_name}")
		df = pd.read_csv(file_name + '.csv', delimiter='|', header=None)
		text = df[5].tolist()
		return text
	except FileNotFoundError:
		print(f"{file_name} not found. Continuing..." )
		return []

def write_text_file(text, file_name):
	with open(file_name, 'w', encoding='utf-8') as txt_file:
		for line in flatten(text):
			if type(line) is float:
				continue
			txt_file.write(line + '\n')

# Process CSV files for training data
train_text = []
train_files_in = [PATH +f"{file}" for file in ['jorf_2023', 'jorf_2022', 'jorf_2019', 'jorf_2018']]
train_file_out = PATH+'training.txt'
for file in train_files_in:
	train_text.append(read_csv_and_extract_text(file))
write_text_file(train_text, train_file_out)

# Process CSV files for testing data
test_text = []
test_files = [PATH +f"{file}" for file in ['jorf_2021', 'jorf_2020']]
test_file_out = PATH+'testing.txt'
for file in test_files:
	test_text.append(read_csv_and_extract_text(file))
write_text_file(test_text, test_file_out)

# Train the data
spm.SentencePieceTrainer.train(input=train_file_out, model_prefix='trained', vocab_size=10000)
sp = spm.SentencePieceProcessor(model_file='trained.model')

# Tokenization of test file
with open(test_file_out, 'r', newline='', encoding='utf-8') as txt_file:
	file_content = txt_file.read()

tokens = sp.encode_as_pieces(file_content.lower())
print("Tokens:", tokens)
