import sentencepiece as spm
import csv

with open('jorf_2023.csv','r', newline='') as csvfile :
    data = csv.reader(csvfile, delimiter='|')
    
    with open('texte.txt','w') as text :
        for line in data:
            text.write(line[5] + "\n")

## Création d'un modèle et entraînement
fichier_texte = 'texte.txt'
modele_sortie = 'modele_v0'
spm.SentencePieceTrainer.Train('--input={} --model_prefix={} --vocab_size=1000'.format(fichier_texte, modele_sortie))

modele = spm.SentencePieceProcessor()
modele.Load('{}.model'.format(modele_sortie))

# Tokenization
fichier_tokens = 'tokens.txt'
with open(fichier_texte, 'r', encoding='utf-8') as file, \
    open(fichier_tokens, 'w', encoding='utf-8') as token_file:
    lines = file.readlines()

    for line in lines:
        tokens = modele.EncodeAsPieces(line.strip())
        line_with_tokens = ' '.join(tokens)  # Convertir la liste de tokens en une chaîne de caractères
        token_file.write(line_with_tokens + '\n')

# Décodage du fichier tokenisé
fichier_phrases = 'token_to_output.txt'

with open(fichier_tokens, 'r', encoding='utf-8') as token_file, \
     open(fichier_phrases, 'w', encoding='utf-8') as output:
    token_lines = token_file.readlines()

    for line in token_lines:
        tokens = line.strip().split(' ')
        phrase_decodee = modele.DecodePieces(tokens)
        output.write(phrase_decodee + '\n')
