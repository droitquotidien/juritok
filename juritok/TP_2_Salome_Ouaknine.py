import pandas as pd
import sentencepiece as spm
"""
1. Importer le trainset, sous la forme d'un csv, séparé par des `|`
- récupérer le texte des articles, contenu dans la 5e colonne, transformer en .txt pour entrainer le tokenizer

"""
table = pd.read_csv('/Users/salomeouaknine/juritok/jorf_2023.csv',sep='|')
table.iloc[:, 5].to_csv('train_JORF.txt', index=False, header=False, sep='\t')

"""
2. Entrainement du tokenizer
"""
spm.SentencePieceTrainer.train('--input=train_JORF.txt --model_prefix=example_model --vocab_size=2000')
sp = spm.SentencePieceProcessor()
sp.load('example_model.model')

"""
3. Test : tokenization d'un article test du JORF du 7 octobre avec notre modèle 
- test (sous le nom de test_1.txt) = "Article 1 

Il est créé un traitement de données à caractère personnel dénommé « répertoire statistique des individus et des logements » (Résil), placé sous la responsabilité du directeur général de l'Institut national de la statistique et des études économiques.
Ce traitement est mis en œuvre pour l'exécution d'une mission d'intérêt public, conformément au e du 1 de l'article 6 du règlement (UE) 2016/679 du Parlement européen et du Conseil du 27 avril 2016 susvisé.
Il a pour finalité, en vue de contribuer au débat public ainsi qu'à l'élaboration et à l'évaluation des politiques publiques, de renforcer la capacité de l'Institut national de la statistique et des études économiques et des services statistiques ministériels à produire des données et études statistiques, en permettant l'établissement d'un répertoire national de la population et des logements et en facilitant les appariements de données administratives avec d'autres sources de données.
Ces appariements constituent des mises en relation, au sens du 3° du I de l'article 33 de la loi du 6 janvier 1978 susvisée, entre les données à caractère personnel enregistrées sur le « répertoire statistique des individus et des logements » et des sources de données statistiques tierces. Ils donnent lieu à la création de nouveaux fichiers, lesquels constituent des traitements de données à caractère personnel au sens du règlement (UE) 2016/679 du Parlement européen et du Conseil du 27 avril 2016 susvisé.
Ce traitement ainsi que ceux résultant des appariements opérés sont établis aux seules fins de production de statistiques publiques, à l'exclusion de toute autre finalité ou de tout autre usage. Ils sont mis en œuvre dans les conditions prévues par la loi du 7 juin 1951 susvisée et dans le respect des règles déontologiques, notamment d'indépendance professionnelle et de respect du secret statistique, applicables à la profession de statisticien."

- ouverture du texte
- tokenization
"""
with open('/Users/salomeouaknine/juritok/test_1.txt', 'r', encoding='utf-8') as file:
    test = file.read()
#print(sp.encode_as_pieces(test))

"""
4. Ne garder que les tokens suffisament fréquents 
- récupérer les tokens en liste (vocabs)
- calculer la fréquence des tokens dans le trainset dans le dictionnaire freq
- ne garder que ceux qui apparaissent au moins 1000 fois dans le trainset
- tester l'encodage alors de notre article test (présent au 3)
"""
vocabs = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]

freq = {}
with open('train_JORF.txt', 'r') as f:
    for line in f:
        line = line.rstrip()
        for piece in sp.encode_as_pieces(line):
            freq.setdefault(piece, 0)
            freq[piece] += 1

vocabs_2 = list(filter(lambda x: x in freq and freq[x] > 1000, vocabs))
sp.set_vocabulary(vocabs_2)
print(sp.encode_as_pieces(test))
