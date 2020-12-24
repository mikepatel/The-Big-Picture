"""
Michael Patel
December 2020

Project description:
    Perform data analysis and visualization of viewing habits

"""
################################################################################
# Imports
import os
import numpy as np
import pandas as pd
import tensorflow as tf


################################################################################
# training
NUM_EPOCHS = 1
BATCH_SIZE = 16

# model
NUM_RNN_UNITS = 128
EMBEDDING_DIM = 100
MAX_WORDS = 1000  # limit vocab size to top X words
MAX_SEQ_LENGTH = 100


################################################################################
# Main
if __name__ == "__main__":
    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    # input: title
    # output: genre

    # read in csv
    csv_filename = "data.csv"
    csv_filepath = os.path.join(os.getcwd(), csv_filename)
    df = pd.read_csv(csv_filepath)

    # create a dataset from dataframe
    dataset = tf.data.Dataset.from_tensor_slices((df["Title"], df["Genre"]))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size=1)

    # preprocessing: standardization, tokenization, vectorization
    preprocess_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        #max_tokens=MAX_WORDS
        #output_sequence_length=MAX_SEQ_LENGTH
    )

    # apply preprocessing to dataset
    train_text = dataset.map(lambda title, genre: title)  # no genre labels
    preprocess_layer.adapt(train_text)
    vocab = np.array(preprocess_layer.get_vocabulary())  # ordered by frequency
    print(vocab[:30])

    quit()
