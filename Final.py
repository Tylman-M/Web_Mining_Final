import pandas as pd, numpy as np, tensorflow as tf,  keras_nlp as nlp, keras


#|%%--%%| <PugefRXIFZ|q2I0VyEDs3>

#df = pd.read_csv('./data/en-fr.csv')
#df.head()


#|%%--%%| <q2I0VyEDs3|pK2JFKStii>


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

