import sentencepiece as spm
import pandas as pd

def read_jorf_files(train_file_name, test_file_name):
    
    file_path_list = ['jorf_20{}.csv'.format(i) for i in range(19,24)]
    
    # Read csv files 
    df_list = []
    for file_path in file_path_list:
        df = pd.read_csv(file_path, sep='|', encoding='utf-8').iloc[:, 5]
        df_list.append(df)

    # We take 2019-2022 as train, 2023 as test
    df_train = pd.concat(df_list[:-1])
    df_test = df_list[-1]

    df_train_list = ''.join(df_train)
    df_test_list = ''.join(df_test)
    
    with open(train_file_name, 'w', encoding='utf-8') as train_file:
        train_file.write(df_train_list)
    with open(test_file_name, 'w', encoding='utf-8') as test_file:
        test_file.write(df_test_list)

def tokenization(train_file_name, prefix, VOC_SIZE, test_file_name):
    spm.SentencePieceTrainer.train('--input={} --model_prefix={} --vocab_size={}'.format(train_file_name, prefix, VOC_SIZE))
    
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('{}.model'.format(prefix))
    
    with open(test_file_name, 'r', encoding='utf-8') as test_file:
        sentence = test_file.readline()
        tokens = tokenizer.EncodeAsPieces(sentence)
        sentence_decoded = tokenizer.DecodePieces(tokens)
        print("sentence:",len(sentence))
        print("tokens:",len(tokens))
        print("sentence_decoded:",len(sentence_decoded))

if __name__ == "__main__":
    read_jorf_files(train_file_name = "train.txt", test_file_name = "test.txt")
    tokenization(train_file_name = "train.txt", prefix = "model", VOC_SIZE = 1000, test_file_name = "test.txt")
    
    
    
    

