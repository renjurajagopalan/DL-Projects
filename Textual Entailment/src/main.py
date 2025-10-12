import pandas as pd
import numpy as np
import import_ipynb
from utils import tokenize, apply_SMOTE, gen_word_embedding
from hypermodels import LSTMHyperModel, GRUHyperModel, CNNHyperModel, SBERTHyperModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

from keras_tuner import BayesianOptimization
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt

import wandb
from wandb.integration.keras import WandbCallback

from sentence_transformers import SentenceTransformer

from config import modelname, wandb_API_KEY

# Read data source
df = pd.read_csv("../data/sententence_data.csv")

# Create a single Target column
df['Category'] =  df['entailment_AB'] + '_' + df['entailment_BA']
X = df[['sentence_A', 'sentence_B']]
y = df['Category']

# Encode the Target Column

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = to_categorical(y)

with open('label_encoder_classes.pkl', 'wb') as f:
    pickle.dump(label_encoder.classes_, f)

# Preprocess the input , generate embedding matrix
def pre_process(modelname):
    if modelname != 'SBERT':
        MAX_NUM_WORDS = 100000
        MAX_SEQUENCE_LENGTH = 50
        EMBEDDING_DIM = 300

        word_index, padded_A, padded_B = tokenize(df, MAX_SEQUENCE_LENGTH, MAX_NUM_WORDS)

        X_combined = np.hstack([padded_A, padded_B])
        X_resampled, y_resampled = apply_SMOTE(X_combined, y)

        X_A_resampled = X_resampled[:, :MAX_SEQUENCE_LENGTH]
        X_B_resampled = X_resampled[:, MAX_SEQUENCE_LENGTH:]

        embedding_matrix = gen_word_embedding(EMBEDDING_DIM, MAX_NUM_WORDS,word_index)

        X_trainA, X_testA, X_trainB, X_testB, y_train, y_test = train_test_split(X_A_resampled, X_B_resampled, y_resampled, test_size=0.2, random_state=42)

        if modelname == 'LSTM':
            selected_model =  LSTMHyperModel(embedding_matrix=embedding_matrix, 
                              MAX_NUM_WORDS=MAX_NUM_WORDS,
                              EMBEDDING_DIM = EMBEDDING_DIM, 
                              MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH)
        
        elif modelname == 'GRU':
            selected_model = GRUHyperModel(embedding_matrix=embedding_matrix, 
                              MAX_NUM_WORDS=MAX_NUM_WORDS,
                              EMBEDDING_DIM = EMBEDDING_DIM, 
                              MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH)
        
        elif modelname == 'CNN':
            selected_model = CNNHyperModel(embedding_matrix=embedding_matrix, 
                              MAX_NUM_WORDS=MAX_NUM_WORDS,
                              EMBEDDING_DIM = EMBEDDING_DIM, 
                              MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH)
        

    else:
        sbert_model_name = 'all-mpnet-base-v2'
        sbert_model = SentenceTransformer(sbert_model_name)

        embeddings_A = sbert_model.encode(df['sentence_A'])
        embeddings_B = sbert_model.encode(df['sentence_B'])
        EMBEDDING_DIM = embeddings_A.shape[1]

        X_combined = np.hstack([embeddings_A, embeddings_B])
        X_resampled, y_resampled = apply_SMOTE(X_combined, y)

        X_A_resampled = X_resampled[:, :EMBEDDING_DIM]
        X_B_resampled = X_resampled[:, EMBEDDING_DIM:]

        X_trainA, X_testA, X_trainB, X_testB, y_train, y_test = train_test_split(X_A_resampled, X_B_resampled, y_resampled, test_size=0.2, random_state=42)
        selected_model = SBERTHyperModel(EMBEDDING_DIM = EMBEDDING_DIM)

    return selected_model, X_trainA, X_testA, X_trainB, X_testB, y_train, y_test


selected_model, X_trainA, X_testA, X_trainB, X_testB, y_train, y_test = pre_process(modelname)

# Wandb Login
wandb.login(key=wandb_API_KEY)

# Keras hypertuner
class MyTuner(kt.Tuner):
    def run_trial(self, trial, X_input, y_train, batch_size, epochs, objective):
        hp = trial.hyperparameters
        objective_name_str = objective

        ## create the model with the current trial hyperparameters
        model = self.hypermodel.build(hp)

        # Initiates new run for each trial on the dashboard of Weights & Biases
        run = wandb.init(project="entailment_classifier_LSTM", config=hp.values)
        print("I am running")

        history = model.fit(X_input,
                  y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.1,
                  callbacks=[WandbCallback()])
        
        training_loss = history.history['loss'][-1] 

        
        self.oracle.update_trial(trial.trial_id, {objective_name_str:training_loss})

        run.finish()

# Hyper parameter Tuning

objective = 'loss' 

'''tuner = BayesianOptimization(
    lstm_model,
    objective='val_accuracy',
    max_trials=30,  # Number of different hyperparameter combinations to try
    directory='hp_tuning_dir',
    project_name='entailment_classifier'
)
'''
tuner = MyTuner(
      oracle=kt.oracles.BayesianOptimizationOracle(
          objective=objective,
          max_trials=4),
      hypermodel=selected_model,
      directory='./hp_tuning_dir')

tuner.search_space_summary()

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)


tuner.search(
    X_input= [X_trainA, X_trainB],
    y_train=y_train,
    validation_data=([X_testA, X_testB], y_test),
    epochs=5,  # Reduced for faster tuning
    batch_size=16,  # Can be larger with SBERT as we don't need to process sequences
    #callbacks=[early_stopping],
    objective=objective
)

# Get the best model hyper parameters and train the model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

run = wandb.init(project = 'Textual Entailment')

# checkpoint
checkpoint_filepath = './saved_models/' + modelname + '.keras'
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)


# Train the selected model
history = best_model.fit(
    [X_trainA, X_trainB],
    y_train,
    validation_data=([X_testA, X_testB], y_test),
    batch_size=32,
    epochs=10,
    callbacks=[early_stopping, model_checkpoint
               ],
    verbose=1
)





