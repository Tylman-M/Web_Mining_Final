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
# WordPiece Tokenizer
ENG_VOCAB_SIZE = 30000
FR_VOCAB_SIZE = 30000

# Preprocessing params.
PRETRAINING_BATCH_SIZE = 128
FINETUNING_BATCH_SIZE = 32
SEQ_LENGTH = 128
MASK_RATE = 0.25
PREDICTIONS_PER_SEQ = 32

# Model params.
NUM_LAYERS = 3
MODEL_DIM = 256
#INTERMEDIATE_DIM = 512
#NUM_HEADS = 4
DROPOUT = 0.1
NORM_EPSILON = 1e-5

EMBED_DIM = 256
INTERMEDIATE_DIM = 2048
NUM_HEADS = 8
MAX_SEQUENCE_LENGTH = 64

# Training params.
PRETRAINING_LEARNING_RATE = 5e-4
PRETRAINING_EPOCHS = 8
FINETUNING_LEARNING_RATE = 5e-5
FINETUNING_EPOCHS = 3


#|%%--%%| <t0TkHKeH9E|7WvQC4Sl1s>

trainfile = './data/smol_train.csv' if dev else './data/train.csv'
testfile = './data/smol_test.csv' if dev else './data/test.csv'
train_data = tf.data.experimental.CsvDataset(trainfile, [tf.string, tf.string], header=True).batch(FINETUNING_BATCH_SIZE)
test_data = tf.data.experimental.CsvDataset(trainfile, [tf.string, tf.string], header=True).batch(FINETUNING_BATCH_SIZE)

#|%%--%%| <7WvQC4Sl1s|a5JxK8rOWB>

print(train_data.unbatch().batch(4).take(1).get_single_element())

#|%%--%%| <a5JxK8rOWB|O9azS94Cqu>

def train_word_piece(text_samples, vocab_size, reserved_tokens):
    word_piece_ds = tf.data.Dataset.from_tensor_slices(text_samples)
    vocab = nlp.tokenizers.compute_word_piece_vocabulary(
        word_piece_ds.batch(1000).prefetch(2),
        vocabulary_size=vocab_size,
        reserved_tokens=reserved_tokens,
    )
    return vocab

reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

eng_vocab = train_word_piece(train['en'], ENG_VOCAB_SIZE, reserved_tokens)
fr_vocab = train_word_piece(train['fr'], FR_VOCAB_SIZE, reserved_tokens)

#|%%--%%| <O9azS94Cqu|QNpknHNU2V>

print(eng_vocab)
print(fr_vocab)

#|%%--%%| <QNpknHNU2V|1MUo3gpuvj>

eng_tokenizer = nlp.tokenizers.WordPieceTokenizer(eng_vocab)
fr_tokenizer = nlp.tokenizers.WordPieceTokenizer(fr_vocab)

#|%%--%%| <1MUo3gpuvj|0kT2Bl0dCs>



def create_model():

    # Encoder
    encoder_inputs = keras.Input(shape=(None,), name="encoder_inputs")
    
    x = nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=ENG_VOCAB_SIZE,
        sequence_length=MAX_SEQUENCE_LENGTH,
        embedding_dim=EMBED_DIM,
    )(encoder_inputs)
    
    encoder_outputs = nlp.layers.TransformerEncoder(
        intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
    )(inputs=x)
    encoder = keras.Model(encoder_inputs, encoder_outputs)
    
    
    # Decoder
    decoder_inputs = keras.Input(shape=(None,), name="decoder_inputs")
    encoded_seq_inputs = keras.Input(shape=(None, EMBED_DIM), name="decoder_state_inputs")
    
    x = nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=FR_VOCAB_SIZE,
        sequence_length=MAX_SEQUENCE_LENGTH,
        embedding_dim=EMBED_DIM,
    )(decoder_inputs)
    
    x = nlp.layers.TransformerDecoder(
        intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
    )(decoder_sequence=x, encoder_sequence=encoded_seq_inputs)
    x = keras.layers.Dropout(0.5)(x)
    decoder_outputs = keras.layers.Dense(FR_VOCAB_SIZE, activation="softmax")(x)
    decoder = keras.Model(
        [
            decoder_inputs,
            encoded_seq_inputs,
        ],
        decoder_outputs,
    )
    decoder_outputs = decoder([decoder_inputs, encoder_outputs])
    
    transformer = keras.Model(
        [encoder_inputs, decoder_inputs],
        decoder_outputs,
        name="transformer",
    )
    return transformer
    
#|%%--%%| <0kT2Bl0dCs|D6vkee0G5w>

transformer = create_model()
transformer.summary()
#|%%--%%| <D6vkee0G5w|C3U4Tmw3OH>
transformer.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
transformer.fit(train_data, epochs=10, validation_data=test_data)

