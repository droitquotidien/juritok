import csv
import re
import sentencepiece as spm

# Put all the data into a text file.
f = open('loi.txt', 'w', encoding='utf-8')

# Let's combine the 4 last years `.csv` and keep one for the test (2019).
for file in ['jorf_2023.csv', 'jorf_2022.csv', 'jorf_2021.csv', 'jorf_2020.csv']:
    with open(file, 'r', newline='', encoding='utf-8') as fichier_csv:
        lecteur_csv = csv.reader(fichier_csv, delimiter='|')
        
        i = 0
        for ligne in lecteur_csv:
            # I chose to keep only the parts of the law and not all the text surrounding.
            pattern = re.compile(r'« ([^«]*?) » sont remplacés par les mots : « ([^«]*?) »')
            matches = pattern.findall(ligne[5])
            if matches:
                for text in matches:
                    f.write(text[0].lower()+"\n")
                    f.write(text[1].lower()+"\n")

f.close()

# Train the data
spm.SentencePieceTrainer.train(input='loi.txt', model_prefix='jorf_2020-2023', vocab_size=1000)

# Test the model
sp = spm.SentencePieceProcessor(model_file='jorf_2020-2023.model')

text = ""

# Tokenization of ten sentences of the last `.csv` to test the model
with open('jorf_2019.csv', 'r', newline='', encoding='utf-8') as fichier_csv:
    lecteur_csv = csv.reader(fichier_csv, delimiter='|')
    i = 0
    for ligne in lecteur_csv:
        pattern = re.compile(r'« ([^«]*?) » sont remplacés par les mots : « ([^«]*?) »')
        matches = pattern.findall(ligne[5])
        if matches and i<=10:
            for tuple in matches:
                text += tuple[0]+"\n"
                text += tuple[1]+"\n"
            i += 1

tokens = sp.encode_as_pieces(text.lower())

# Print the tokens
print("Phrase originale:", text)
print("Tokens:", tokens)

# Tokens decoding
phrase_reconstruite = sp.decode_pieces(tokens)
print("Phrase reconstruite:", phrase_reconstruite)