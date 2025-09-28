# %%
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack

import pandas as pd
import numpy as np

import re
import nltk

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.preprocessing.sequence import pad_sequences

from pathlib import Path

# %% [markdown]
# ### Pre-process the input text by removing the characters

# %%
def clean_text(text):
  # pre-process the input text

    text = str(text)

    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)

    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    text = REPLACE_BY_SPACE_RE.sub(' ', text)

    text = text.lower()

    return text

# %% [markdown]
# ### Tokenize and apply padding

# %%
def tokenize(df, MAX_SEQUENCE_LENGTH, MAX_NUM_WORDS):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    sentences = list(df['sentence_A']) + list(df['sentence_B'])

    tokenizer.fit_on_texts(sentences)
    sequences_A = tokenizer.texts_to_sequences(df['sentence_A'])

    #sequences_A = vectorizer(df['sentence_A'])
    #sequences_B = vectorizer(df['sentence_B'])
    sequences_B = tokenizer.texts_to_sequences(df['sentence_B'])

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    padded_A = pad_sequences(sequences_A, maxlen=MAX_SEQUENCE_LENGTH)
    padded_B = pad_sequences(sequences_B, maxlen=MAX_SEQUENCE_LENGTH)

    return word_index, padded_A, padded_B


# %% [markdown]
# ### Apply SMOTE 

# %%
def apply_SMOTE(X_combined, y):
    smote = SMOTE(random_state = 42)
    X_combined_resampled, y_resampled = smote.fit_resample(X_combined, y)
    return X_combined_resampled, y_resampled

# %% [markdown]
# ### Generate Word Embeddings 

# %%
def gen_word_embedding(EMBEDDING_DIM, MAX_NUM_WORDS,word_index):
    f = open('../word_embedding/glove.6B.300d.txt', encoding="utf8")
    embeddings_index = {}
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    embedding_matrix = np.zeros((MAX_NUM_WORDS, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i < MAX_NUM_WORDS:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

# %% [markdown]
# ### Tokenize Function for Test Inputs

# %%
def tokenize_test(sentence1, sentence2):

    print(sentence1, " ", sentence2)
    MAX_SEQUENCE_LENGTH = 50
    tokenizer = Tokenizer(num_words=100000)
    
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent
    file_path = script_dir / 'data' / 'sententence_data.csv'

    df = pd.read_csv(file_path)
    sentences = list(df['sentence_A']) + list(df['sentence_B'])

    tokenizer.fit_on_texts(sentences)
    sequences_A = tokenizer.texts_to_sequences([sentence1])
    sequences_B = tokenizer.texts_to_sequences([sentence2])
  
    padded_A = pad_sequences(sequences_A, maxlen=MAX_SEQUENCE_LENGTH)
    padded_B = pad_sequences(sequences_B, maxlen=MAX_SEQUENCE_LENGTH)

    return padded_A, padded_B


