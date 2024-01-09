import pandas as pd
import sentencepiece as spm


JO_file = pd.read_csv('juritok/jorf_2023.csv',header=None,sep='|')

# Prepare the corpus
def csv2txt(csv):
    text_unique = csv[0].unique()
    corpus = []

    for indice,text in enumerate(text_unique):
        relative_data = csv[csv[0] == text]
        contents = ""
        print(indice)

        for i in range(relative_data.shape[0]):
            contents += relative_data.iloc[i,-1]
            contents += " "
        
        corpus += [contents]

    res = "\n".join(corpus)
    return res

res = csv2txt(JO_file)
with open('corpus.txt','w') as file:
    file.write(res)

# Train the model
spm.SentencePieceTrainer.train(input='corpus.txt', model_prefix='model', vocab_size=5000)
token_model = spm.SentencePieceProcessor(model_file='model.model')

# Apply the model to the JO_csv and get a column of tokens
JO_file["int_token"] = JO_file[5].apply(token_model.encode)
JO_file["str_token"] = JO_file[5].apply(token_model.encode_as_pieces)

# Give an exemple of tokenized file
JO_file.iloc[:100,:].to_csv("juritok/tokenized_jorf_2023.csv")