import sentencepiece as spm

# paths
train_file = 'jorf.txt'
model_prefix = 'modele'

# parameters
vocab_size = 10000
model_type = 'bpe'
character_coverage = 1.0

# training
spm.SentencePieceTrainer.train(
    input=train_file,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    model_type=model_type,
    character_coverage=character_coverage
)

# load model
sp = spm.SentencePieceProcessor()
sp.load(f'{model_prefix}.model')

# testing
text = "JORFTEXT000039696471|JORFARTI000039696492|1|3|4|2° Au premier alinéa de l'article R. 821-5 :"
tokens = sp.encode_as_pieces(text)
print(tokens)
