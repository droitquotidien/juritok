import pandas as pd
import sentencepiece as spm
import re

# Read and import the data
file = 'jorf_2023.csv'
file2 = 'jorf_2022.csv'
file3 = 'jorf_2021.csv'
file4 = 'jorf_2020.csv'
file5 = 'jorf_2019.csv'
df1 = pd.read_csv(file, sep='|')
df2 = pd.read_csv(file2, sep='|')
df3 = pd.read_csv(file3, sep='|') # used for testing
df4 = pd.read_csv(file4, sep='|')
df5 = pd.read_csv(file5, sep='|')

# merge the dataframes
df = pd.concat([df1, df2, df4, df5], ignore_index=True)

# The text is in the last column (-1)
text_column = df.iloc[:, -1]

# Convert the DataFrame column to string
text = text_column.astype(str)

# remove any of the following characters: :, >, <, ;, ., -, _, =, +, *, /, (, ), [, ], {, }, ', \, %, !, ?, $, #, @, &, ~, `, ^, %
text = text.str.replace(r'[:,><;.\-_=+*/()\[\]{}\'\\%!?\$#@&~`^%]', '')

# Save text to a temporary file
text_file = 'temp.txt'
text.to_csv(text_file, header=False, index=False)

# Train SentencePiece model, vocab_size is the number of unique tokens
spm.SentencePieceTrainer.train(input=text_file, model_prefix='spm_model', vocab_size=11000)

# Load SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load('spm_model.model')

# Tokenize the first 10 rows of the data (not trained on)
for i in df3.iloc[:,-1].head(10):
    # remove any of the following characters: :, >, <, ;, ., -, _, =, +, *, /, (, ), [, ], {, }, ', \, %, !, ?, $, #, @, &, ~, `, ^, %
    i = re.sub(r'[:,><;.\-_=+*/()\[\]{}\'\\%!?\$#@&~`^%]', '', i)
    
    # tokenize
    tokens = sp.encode_as_pieces(i)
    print("Number of tokens", len(tokens))
    print("The tokens:", tokens)
    print()