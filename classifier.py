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

BUFFER_SIZE = 10000
MAX_WORDS = 1000  # limit vocab size to top X words
MAX_SEQ_LENGTH = 100

# model
NUM_RNN_UNITS = 128
EMBEDDING_DIM = 100



################################################################################
# Main
if __name__ == "__main__":
    # input: title
    # output: genre

    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    # read in csv
    csv_filename = "data.csv"
    csv_filepath = os.path.join(os.getcwd(), csv_filename)
    df = pd.read_csv(csv_filepath)

    # features: titles
    titles = df["Title"]

    # labels: genres
    # map genre labels to integers
    genres = df["Genre"].unique()
    genres = sorted(genres)
    num_genres = len(genres)
    genre2int = {v: k for k, v in enumerate(genres)}
    int2genre = {k: v for k, v in enumerate(genres)}

    genres = [genre2int[i] for i in df["Genre"]]

    # create a dataset from dataframe
    dataset = tf.data.Dataset.from_tensor_slices((titles, genres))
    dataset = dataset.shuffle(buffer_size=BUFFER_SIZE)
    dataset = dataset.batch(batch_size=BATCH_SIZE)

    # preprocessing: standardization, tokenization, vectorization
    preprocess_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        #max_tokens=MAX_WORDS
        #output_sequence_length=MAX_SEQ_LENGTH
    )

    # apply preprocessing to dataset
    train_text = dataset.map(lambda title, genre: title)  # titles
    train_labels = dataset.map(lambda title, genre: genre)  # genres
    preprocess_layer.adapt(train_text)
    vocab = preprocess_layer.get_vocabulary()  # ordered by frequency
    vocab_size = len(vocab)

    def vectorize_text(t, g):
        t = tf.expand_dims(t, axis=-1)
        return preprocess_layer(t), g

    train_dataset = dataset.map(vectorize_text)

    # data pipeline performance: cache and prefetch
    train_dataset = train_dataset.cache().prefetch(buffer_size=BUFFER_SIZE)

    # ----- MODEL ----- #
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(
        input_dim=vocab_size+2,  # vocab size
        output_dim=EMBEDDING_DIM
    ))  # (batch, sequence, embedding)

    model.add(tf.keras.layers.GRU(
        units=NUM_RNN_UNITS
    ))

    model.add(tf.keras.layers.Dense(
        units=num_genres,
        activation=tf.keras.activations.softmax
    ))

    model.summary()

    # loss function and optimizer
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    # train model
    history = model.fit(
        x=train_dataset,
        epochs=NUM_EPOCHS
    )

    # ----- PREDICTION ----- #
    test_titles = [
        "Harry Potter and the Prisoner of Azkaban",
        "Pulp Fiction",
        "Jaws",
        "Chinatown",
        "Anchorman",
        "The Matrix",
        "Titanic",
        "The English Patient",
        "Power Rangers",
        "Forrest Gump",
        "Spiderman 2",
        "A Nightmare on Elm Street",
        "Friday the 13th"
    ]

    # preprocess
    test_dataset = np.expand_dims(test_titles, axis=-1)
    test_dataset = preprocess_layer(test_dataset)

    # make predictions
    predictions = model.predict(test_dataset)

    # print out predictions
    for i in range(len(predictions)):
        print()
        print(f'Title: {test_titles[i]}')
        print(f'Genre: {int2genre[np.argmax(predictions[i])]}')

    quit()
