import pandas as pd, numpy as np, tensorflow as tf,  keras_nlp as nlp, keras
from sklearn.model_selection import train_test_split


#|%%--%%| <PugefRXIFZ|S3vlxen8zo>
#Generate split
generate_split = False
dev = True
if generate_split:
    df = pd.read_csv('./data/en-fr.csv', encoding='latin-1')
    train, test = train_test_split(df, test_size=0.2)
    train, test = pd.DataFrame(train), pd.DataFrame(test) #My LSP is bugged, this shut up the warnings
    train.to_csv('./data/train.csv', index=False)
    train.iloc[:10000].to_csv('./data/smol_train.csv', index=False)
    test.to_csv('./data/test.csv', index=False)
    test.iloc[:1000].to_csv('./data/smol_test.csv', index=False)
    if dev:
        train = train.iloc[:10000]
        test = test.iloc[:1000]
else:
    if dev:
        train = pd.read_csv('./data/smol_train.csv', encoding='latin-1')
        test = pd.read_csv('./data/smol_test.csv', encoding='latin-1')
    else:
        train = pd.read_csv('./data/train.csv', encoding='latin-1')
        test = pd.read_csv('./data/test.csv', encoding='latin-1')
print(train.shape, test.shape)
print(train.head())
print(test.head())

#|%%--%%| <S3vlxen8zo|KqiTcyJspM>


#|%%--%%| <KqiTcyJspM|pK2JFKStii>

#|%%--%%| <pK2JFKStii|t0TkHKeH9E>
# Examples from kerasnlp
# Preprocessing params.
PRETRAINING_BATCH_SIZE = 128
FINETUNING_BATCH_SIZE = 32
SEQ_LENGTH = 128
MASK_RATE = 0.25
PREDICTIONS_PER_SEQ = 32

# Model params.
NUM_LAYERS = 3
MODEL_DIM = 256
INTERMEDIATE_DIM = 512
NUM_HEADS = 4
DROPOUT = 0.1
NORM_EPSILON = 1e-5

# Training params.
PRETRAINING_LEARNING_RATE = 5e-4
PRETRAINING_EPOCHS = 8
FINETUNING_LEARNING_RATE = 5e-5
FINETUNING_EPOCHS = 3

#|%%--%%| <t0TkHKeH9E|7WvQC4Sl1s>

data = tf.data.experimental.CsvDataset('./data/en-fr.csv', [tf.string, tf.string], header=True).batch(FINETUNING_BATCH_SIZE) 
#|%%--%%| <7WvQC4Sl1s|a5JxK8rOWB>

print(data.unbatch().batch(4).take(1).get_single_element())

#|%%--%%| <a5JxK8rOWB|O9azS94Cqu>

tokenizer = nlp.tokenizers.WordPieceTokenizer()

