import sentencepiece as spm

# First run write_txt.py to create jorf_20**.txt

# Train the data (we could use more than juste jorf_2023.txt)
# I tried different vocab sizes and 10 000 seems to be the most efficient to train the model
# because with a smaller size tokens are just letters which is not ideal
spm.SentencePieceTrainer.train(input='jorf_2023.txt', model_prefix='jorf_2023', vocab_size=10000)

# Test the model
sp = spm.SentencePieceProcessor(model_file='jorf_2023.model')

# Tokenization of other text file to test
with open('jorf_2022.txt', 'r', newline='', encoding='utf-8') as txtFile:
    file = txtFile.read()

tokens = sp.encode_as_pieces(file.lower())

# Print the tokens
print("Original file :", file)
print("Tokens :", tokens)

# Tokens decoding
reconstitute_file = sp.decode_pieces(tokens)
print("Reconstitute file :", reconstitute_file)