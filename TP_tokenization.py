import re
import csv
import sentencepiece as spm

#extracting laws and text only into a .txt file

f = open('laws_2020_2022.txt', 'w', encoding='utf-8')

for year in range(2020, 2023):
    with open(f'jorf_{year}.csv', 'r', newline='', encoding='utf-8') as csv_law_file:
        csv_law_reader = csv.reader(csv_law_file, delimiter='|')
        
        remove_unnecessary_lines = re.compile("fr\/.*20\d\d-\d\d-\d\d")
        for line in csv_law_reader:
            if not remove_unnecessary_lines.findall(line[5]):
                f.write(line[5].lower() + "\n")

f.close()

#Training the model
spm.SentencePieceTrainer.train(input="laws_2020_2022.txt", model_prefix='reduced_JO_tokenization', vocab_size=1000, max_sentence_length=50000)

# Charging the model
sp = spm.SentencePieceProcessor(model_file='reduced_JO_tokenization.model')

# Test with a few sentences from JO 2023
test_sentences = []
with open(f'jorf_2023.csv', 'r', newline='', encoding='utf-8') as test_file:
        test_reader = csv.reader(test_file, delimiter='|')
        remove_unnecessary_lines = re.compile("fr\/.*20\d\d-\d\d-\d\d")

        for i, line in enumerate(test_reader):
            if i > 10:
                 break
            elif not remove_unnecessary_lines.findall(line[5]):
                test_sentences.append(line[5].lower() + "\n")
                
tokens = sp.encode_as_pieces(test_sentences)

# Encoding
print("Original sentence :", test_sentences)
print("Tokens:", tokens)

# Decoding
built_sentences = sp.decode_pieces(tokens)
print("Phrase reconstruite:", built_sentences)