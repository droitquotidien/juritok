import sentencepiece as spm
import csv

with open('jorf_2022.csv','r', newline='') as csvfile :
    data = csv.reader(csvfile, delimiter='|')
    
    with open('texte2.txt','w') as text :
        for line in data:
            text.write(line[5] + "\n")

fichier_texte = 'texte2.txt'
modele_sortie = 'modele_v0'

modele = spm.SentencePieceProcessor()
modele.Load('{}.model'.format(modele_sortie))

# Tokenization
fichier_tokens = 'tokens2.txt'
with open(fichier_texte, 'r', encoding='utf-8') as file, \
    open(fichier_tokens, 'w', encoding='utf-8') as token_file:
    lines = file.readlines()

    for line in lines:
        tokens = modele.EncodeAsPieces(line.strip())
        line_with_tokens = ' '.join(tokens)  # Convertir la liste de tokens en une chaîne de caractères
        token_file.write(line_with_tokens + '\n')

# Décodage du fichier tokenisé
fichier_phrases = 'token_to_output2.txt'

with open(fichier_tokens, 'r', encoding='utf-8') as token_file, \
     open(fichier_phrases, 'w', encoding='utf-8') as output:
    token_lines = token_file.readlines()

    for line in token_lines:
        tokens = line.strip().split(' ')
        phrase_decodee = modele.DecodePieces(tokens)
        output.write(phrase_decodee + '\n')
