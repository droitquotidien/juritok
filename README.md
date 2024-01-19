# juritok
Tokenisation des textes du JO et des textes consolidés

# data
Contient les fichiers CSV avec le journal officiel de 2022 et de 2023
Contient les fichiers txt obtenus en sortie du script generate_txt_file.py pour 2022 et 2023

# generate_txt_file
Génère un fichier txt contenant uniquement le texte de droit (entre guillemets) à partir d'un fichier csv tel que ceux auxquels on a accès pour chaque année de journal officiel
Pour exécuter ce script, il faut entrer la commande suivante :
`python -m generate_txt_file --csv_path my_csv_path --txt_path my_txt_path`
où `my_csv_path` est le chemin d'accès au fichier CSV contenant le journal officiel et `my_txt_path` est le chemin d'accès au fichier txt que l'on souhaite créer

# tokenizer.py
Entraîne un tokenizer sur un corpus du journal officiel et stocke le modèle ainsi entraîné
Pour exécuter ce script, il faut entrer la commande suivante :
`python -m generate_txt_file --data my_txt_path --vocab_size my_vocab_size`
où `my_txt_path` est le chemin d'accès au fichier txt que l'on souhaite utiliser pour entraîner le tokenizer et `my_vocab_size` est le nombre de mots qu'on souhaite inclure dans le dictionnaire appris par notre tokenizer

# test.ipynb
Notebook dans lequel on teste le tokenizer m.model (qui a été entraîné sur le journal officiel de 2023) sur des textes juridiques issus du journal officiel de 2022 et on vérifie que l'on retrouve bien le message d'origine lorsqu'on encode puis décode le message d'origine en utilisant ce tokenizer
