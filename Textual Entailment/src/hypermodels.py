# %%
from keras_tuner import HyperModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, GRU,Conv1D, MaxPooling1D,Dense, Input, BatchNormalization, Dropout, Concatenate, Subtract, Multiply,Embedding,Bidirectional,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD

# %%
class LSTMHyperModel(HyperModel):
    def __init__(self,embedding_matrix, MAX_NUM_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH):
        self.embedding_matrix = embedding_matrix
        self.MAX_NUM_WORDS = MAX_NUM_WORDS
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        
    
    def build(self, hp):
        common_embed = Embedding(self.MAX_NUM_WORDS, self.EMBEDDING_DIM, weights=[self.embedding_matrix], input_length=self.MAX_SEQUENCE_LENGTH, trainable=False)

        input_A = Input(shape=(self.MAX_SEQUENCE_LENGTH,))
        input_B = Input(shape=(self.MAX_SEQUENCE_LENGTH,))

        lstm_1 = common_embed(input_A)
        lstm_2 = common_embed(input_B)

        common_lstm1 = Bidirectional(LSTM(hp.Int(
                    "units", min_value=32, max_value=512, step=32, default=128
                ), return_sequences=True, activation = 'relu'))
        lstm1_out = common_lstm1(lstm_1)
        lstm2_out = common_lstm1(lstm_2)

        common_lstm = Bidirectional(LSTM(64, return_sequences=True, activation = 'relu'))
        Vector1 = common_lstm(lstm1_out)
        Vector1 = Flatten()(Vector1)

        Vector2 = common_lstm(lstm2_out)
        Vector2 = Flatten()(Vector2)

        x3 = Subtract()([Vector1, Vector2])
        x3 = Multiply()([x3, x3])

        x1_ = Multiply()([Vector1, Vector1])
        x2_ = Multiply()([Vector2, Vector2])
        x4 = Subtract()([x1_, x2_])

        conc = Concatenate(axis=-1)([x3, x4])

        x = Dense(100, activation='relu')(conc)
        x = BatchNormalization()(x)
        x = Dropout(rate=hp.Float(
                    "dropout_1", min_value=0.0, max_value=0.5, default=0.25, step=0.05,
                ))(x)

        output = Dense(7, activation='softmax')(x)

        model = Model(inputs=[input_A, input_B], outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling="LOG",
                    default=1e-3,
                )), metrics=['accuracy'])

        return model

# %%
class GRUHyperModel(HyperModel):
    def __init__(self,embedding_matrix, MAX_NUM_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH):
        self.embedding_matrix = embedding_matrix
        self.MAX_NUM_WORDS = MAX_NUM_WORDS
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        
    
    def build(self, hp):
        common_embed = Embedding(self.MAX_NUM_WORDS, self.EMBEDDING_DIM, weights=[self.embedding_matrix], input_length=self.MAX_SEQUENCE_LENGTH, trainable=False)

        input_A = Input(shape=(self.MAX_SEQUENCE_LENGTH,))
        input_B = Input(shape=(self.MAX_SEQUENCE_LENGTH,))

        lstm_1 = common_embed(input_A)
        lstm_2 = common_embed(input_B)

        common_lstm1 = Bidirectional(GRU(hp.Int(
                    "units", min_value=32, max_value=512, step=32, default=128
                ), return_sequences=True, activation = 'relu'))
        lstm1_out = common_lstm1(lstm_1)
        lstm2_out = common_lstm1(lstm_2)

        common_lstm = Bidirectional(GRU(64, return_sequences=True, activation = 'relu'))
        Vector1 = common_lstm(lstm1_out)
        Vector1 = Flatten()(Vector1)

        Vector2 = common_lstm(lstm2_out)
        Vector2 = Flatten()(Vector2)

        x3 = Subtract()([Vector1, Vector2])
        x3 = Multiply()([x3, x3])

        x1_ = Multiply()([Vector1, Vector1])
        x2_ = Multiply()([Vector2, Vector2])
        x4 = Subtract()([x1_, x2_])

        conc = Concatenate(axis=-1)([x3, x4])

        x = Dense(100, activation='relu')(conc)
        x = BatchNormalization()(x)
        x = Dropout(rate=hp.Float(
                    "dropout_1", min_value=0.0, max_value=0.5, default=0.25, step=0.05,
                ))(x)

        output = Dense(7, activation='softmax')(x)

        model = Model(inputs=[input_A, input_B], outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling="LOG",
                    default=1e-3,
                )), metrics=['accuracy'])

        return model

# %%
class CNNHyperModel(HyperModel):
    def __init__(self,embedding_matrix, MAX_NUM_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH):
        self.embedding_matrix = embedding_matrix
        self.MAX_NUM_WORDS = MAX_NUM_WORDS
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        
    
    def build(self, hp):
        common_embed = Embedding(self.MAX_NUM_WORDS, self.EMBEDDING_DIM, weights=[self.embedding_matrix], input_length=self.MAX_SEQUENCE_LENGTH, trainable=False)

        input_A = Input(shape=(self.MAX_SEQUENCE_LENGTH,))
        input_B = Input(shape=(self.MAX_SEQUENCE_LENGTH,))

        lstm_1 = common_embed(input_A)
        lstm_2 = common_embed(input_B)

        common_conv1 = Conv1D(filters, kernel_size, padding = 'valid', activation = 'relu')
        lstm1_out = common_conv1(lstm_1)
        lstm2_out = common_conv1(lstm_2)

        maxpool = MaxPooling1D()
        conv1_out = maxpool(lstm1_out)
        conv2_out = maxpool(lstm2_out)

        common_conv2 = Conv1D(filters, kernel_size, padding = 'valid', activation = 'relu')
        Vector1 = common_conv2(conv1_out)
        Vector1 = Flatten()(Vector1)

        Vector2 = common_conv2(conv2_out)
        Vector2 = Flatten()(Vector2)

        x3 = Subtract()([Vector1, Vector2])
        x3 = Multiply()([x3, x3])

        x1_ = Multiply()([Vector1, Vector1])
        x2_ = Multiply()([Vector2, Vector2])
        x4 = Subtract()([x1_, x2_])

        conc = Concatenate(axis=-1)([x3, x4])

        x = Dense(100, activation='relu')(conc)
        x = BatchNormalization()(x)
        x = Dropout(rate=hp.Float(
                    "dropout_1", min_value=0.0, max_value=0.5, default=0.25, step=0.05,
                ))(x)

        output = Dense(7, activation='softmax')(x)

        model = Model(inputs=[input_A, input_B], outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling="LOG",
                    default=1e-3,
                )), metrics=['accuracy'])

        return model

# %%
class SBERTHyperModel(HyperModel):
    def __init__(self,EMBEDDING_DIM):
        self.EMBEDDING_DIM = EMBEDDING_DIM
                
    
    def build(self, hp):
        
        input_A = Input(shape=(self.EMBEDDING_DIM,))
        input_B = Input(shape=(self.EMBEDDING_DIM,))

        diff = Subtract()([input_A, input_B])
        prod = Multiply()([input_A, input_B])

        combined = Concatenate()([input_A, input_B, diff, prod])
        combined = BatchNormalization()(combined)

        x = Dense(hp.Int('first_dense_units', min_value=100, max_value=150, step=10), activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dropout(rate=hp.Float(
                    "dropout_1", min_value=0.0, max_value=0.5, default=0.25, step=0.05,
                ))(x)

        output = Dense(7, activation='softmax')(x)

        model = Model(inputs=[input_A, input_B], outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling="LOG",
                    default=1e-3,
                )), metrics=['accuracy'])

        return model
        



