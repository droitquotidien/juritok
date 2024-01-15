# README: TP Tokenization - François Mazé

## Contenu du dépôt
Dans le dossier `juritok/juritok`, quatre scripts Python :
- `data_pre_processing.py` : crée des fichiers .txt contenant les données d'entraînement et de test à la fois dans le cas où on traite toutes les données (all) et dans le cas où on ne traite que le droit et pas le texte correspondant aux patchs (law)
- `training.py` : crée le modèle de tokenization et l'entraîne sur les données d'entraînement. Crée deux modèles correspondant à all et law
- `test.py` : permet de tester les modèles obtenus

## Fonctionnement des codes

### Etape de preprocesing
Cette étape consiste à lire les fichiers .csv contenant les données (ces fichiers .csv doivent être dans un même dossier, et doivent être les seuls éléments du dossier). Le chemin vers ce dossier doit être indiqué en utilisant la balise `--data_folder` lors de l'appel du script.

Le code génère quatre fichiers .txt correspondant aux fichiers d'entraînement et de test pour les deux cas mentionnés ci-dessus: all (tout le texte est considéré) et law (seul le droit, pas les patchs). Le fichier de test consiste en le dernier fichier (données JORF 2023) et le fichier d'entraînement de tous les autres .csv.

A la suite de la génération des fichiers texte, le code analyse les données d'entraînement pour déterminer la taille de vocabulaire pertinente. On détermine ainsi le nombre de mots nécessaires pour assurer une couverture à un certain niveau. Ce niveau peut être indiqué avec la balise `--vocab_coverage` lors de l'appel.

### Etape d'entraînement
Cette étape crée les modèles (all et law) et les entraîne sur les deux jeux de données d'entraînement. A l'aide des balises `--full_data_file` et `--law_data_file`, on précise la localisation des fichiers d'entraînement. A l'aide des balises, `--full_data_vocab_size` et `--law_data_vocab_size`, on donne la taille de vocabulaire souhaitée (par défaut mise au niveau déterminé par l'étape de preprocessing).

### Etape de test
Cette étape charge les modèles et affiche leurs caractéristiques (nombre de tokens) pour vérifier que le vocab size a été respecté. Les modèles doivent être chargés en utilisant les balises `--full_model` et `--law_model`.

Ensuite, les deux modèles sont testés sur les deux fichiers de test. On tokenize l'intégralité du texte de test, qui doit être indiqué avec les balises `--test_file_full` et `--test_file_law`.

A la suite de la tokenization de tout le texte, on affiche des exemples de phrases tokenizées, on calcule l'usage rate, qui correspond au nombre de tokens utilisés sur le nombre de tokens totaux (env. 93% dans les exemples testés) et on affiche les tokens les plus fréquents (de plus de 5 caractères pour éliminer les tokens triviaux (de, le, etc.)).