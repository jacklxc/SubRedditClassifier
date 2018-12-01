import string
import re
import numpy as np
import tensorflow as tf

sess = tf.Session()

import keras
from keras import optimizers
from keras import backend as K
K.set_session(sess)

from keras import regularizers
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout, Flatten, LSTM, Bidirectional, Lambda, Input
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras import regularizers
from keras.utils import plot_model
from keras.optimizers import RMSprop, Adam
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from tqdm import tqdm
import codecs

def reset_random_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)

class SpreadsheetClassificationExecution:

    def __init__(self, sd, embedding_matrix, classifier_type) :

        #training params
        batch_size = 256
        num_epochs = 100

        #model parameters
        num_filters = 64
        embed_dim = 100
        weight_decay = 1e-4

        if classifier_type == 'SimpleLSTM':
            classifier = SimpleLSTM(embedding_matrix, sd.max_seq_len, sd.n_classes)
        elif classifier_type == 'SimpleCNN':
            classifier = SimpleCNN(embedding_matrix, sd.max_seq_len, sd.n_classes)
        elif classifier_type == 'BiLSTM':
            classifier = BiLSTM(embedding_matrix, sd.max_seq_len, sd.n_classes)
        elif classifier_type == 'CNNBiLSTM':
            classifier = CNNBiLSTM(embedding_matrix, sd.max_seq_len, sd.n_classes)
        else:
            raise ValueError("Incorrect Classifier Type: %s"%(classifier_type))

        model = classifier.model
        print("Begin training")
        #callbacks = [TensorBoard(log_dir='./logs/elmo_hub')]
        #early_stopping = EarlyStopping(patience = 10)
        hist = model.fit(sd.x_train, sd.y_train, batch_size=batch_size,
                         epochs=num_epochs, validation_split=0.1,
                         shuffle=True, verbose=2)
        
        score = model.evaluate(sd.x_test, sd.y_test, verbose=2)
        self.prediction = model.predict(sd.x_test)
        self.loss = score[0]
        self.accuracy = score[1]
        print("Test accuracy:",score[1])

class SpreadsheetData:

    def __init__(self, datafile, labelfile, testSize, vocab, limit, randomize=False, MAX_NUM_WORDS = 400000):
        raw_doc = self.read_txt(datafile,limit)
        raw_label = self.read_txt(labelfile,limit)
        n_rec = len(raw_doc)
        
        raw_doc_np = np.array(raw_doc)
        raw_label_np = np.array(raw_label)

        if randomize:
            self.randomized = True
            indices = np.random.permutation(n_rec)
        else:
            self.randomized = False
            indices = np.arange(n_rec)

        test_indices = indices < testSize
        train_indices = np.logical_not(test_indices)
        train_raw_doc = raw_doc_np[train_indices].tolist()
        test_raw_doc = raw_doc_np[test_indices].tolist()
        train_label = raw_label_np[train_indices].tolist()
        test_label = raw_label_np[test_indices].tolist()

        self.labels = list(set(raw_label_np.tolist()))
        self.n_classes = len(self.labels)

        y_train_base = [self.labels.index(i) for i in train_label]
        y_test_base = [self.labels.index(i) for i in test_label]

        self.max_seq_len = np.max([len(seq.split()) for seq in raw_doc])

        print("pre-processing train data...")
                
        processed_docs_train = []
        for doc in train_raw_doc:
            filtered = []
            tokens = doc.split()
            for word in tokens:
                word = self._clean_url(word)
                word = self._clean_num(word)
                if word not in vocab:
                    word = "<UNK>"
                filtered.append(word)
            processed_docs_train.append(" ".join(filtered))

        processed_docs_test = []
        for doc in test_raw_doc:
            filtered = []
            tokens = doc.split()
            for word in tokens:
                word = self._clean_url(word)
                word = self._clean_num(word)
                if word not in vocab:
                    word = "<UNK>"
                filtered.append(word)
            processed_docs_test.append(" ".join(filtered))

        print("tokenizing input data...")
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, lower=True, char_level=False)
        tokenizer.fit_on_texts(processed_docs_train + processed_docs_test)  #leaky
        word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)
        word_seq_test = tokenizer.texts_to_sequences(processed_docs_test)
        self.word_index = tokenizer.word_index
        print("dictionary size: ", len(self.word_index))

        #pad sequences
        self.x_train = sequence.pad_sequences(word_seq_train, maxlen=self.max_seq_len)
        self.x_test = sequence.pad_sequences(word_seq_test, maxlen=self.max_seq_len)

        self.y_train = keras.utils.to_categorical(y_train_base, num_classes=self.n_classes)
        self.y_test = keras.utils.to_categorical(y_test_base, num_classes=self.n_classes)

    def read_txt(self, filename, limit=None):
        count = 0
        raw_docs = []
        with codecs.open(filename, "r", "utf-8") as f:
            for line in f:
                raw_docs.append(line.strip())
                count += 1
                if limit is not None and count > limit:
                    break
        return raw_docs

    def _clean_url(self,word):
        """
            Clean specific data format from social media
        """
        # clean urls
        word = re.sub(r'https? : \/\/.*[\r\n]*', '<URL>', word)
        word = re.sub(r'exlink', '<URL>', word)
        return word


    def _clean_num(self,word):
        # check if the word contain number and no letters
        if any(char.isdigit() for char in word):
            try:
                num = float(word.replace(',', ''))
                return '<NUM>'
            except:
                if not any(char.isalpha() for char in word):
                    return '<NUM>'
        return word
    
class SimpleCNN:

    rep_max = -100000.0
    rep_size = 0

    def __init__(self, embedding_matrix, max_seq_len, n_classes, num_filters = 64, weight_decay = 1e-4):

        nb_words = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]

        self.model = Sequential()
        self.model.add(Embedding(nb_words, embed_dim,
            weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
        self.model.add(Dropout(0.5))
        self.model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
        self.model.add(MaxPooling1D(2))
        self.model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Dense(n_classes, activation='softmax'))  #multi-label (k-hot encoding)

        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        self.model.summary()

        
class SimpleLSTM:

    def __init__(self, embedding_matrix, max_seq_len, n_classes):

        nb_words = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]

        self.model = Sequential()
        self.model.add(Embedding(nb_words, embed_dim,
                            weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(128))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(n_classes, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                      optimizer="adam",
                      metrics=['accuracy'])
        self.model.summary()

class BiLSTM:
    def __init__(self, embedding_matrix, max_seq_len, n_classes):
        nb_words = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]

        self.model = Sequential()
        self.model.add(Embedding(nb_words, embed_dim,
                                 weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
        self.model.add(Dropout(0.5))
        self.model.add(Bidirectional(LSTM(128)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(n_classes, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        self.model.summary()

class CNNBiLSTM:
    def __init__(self, embedding_matrix, max_seq_len, n_classes, num_filters = 128, weight_decay = 1e-4):

        reg = 1e-4
        dropout = 0.4
        nb_words = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]

        self.model = Sequential()
        self.model.add(Embedding(nb_words, embed_dim,
            weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
        self.model.add(Dropout(dropout))
        self.model.add(Conv1D(num_filters, 7, strides=1, activation='relu', padding='valid',\
                              kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg)))
        self.model.add(Dropout(dropout))
        self.model.add(MaxPooling1D(2))
        self.model.add(Conv1D(num_filters, 7, strides=1, activation='relu', padding='valid',\
                              kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg)))
        self.model.add(Dropout(dropout))
        self.model.add(Bidirectional(LSTM(512,kernel_regularizer=regularizers.l2(reg),\
                                  recurrent_regularizer=regularizers.l2(reg), \
                                  bias_regularizer=regularizers.l2(reg))))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(n_classes, activation='softmax'))  #multi-label (k-hot encoding)
        lr = 0.001
        decay = lr / 100
        adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        self.model.summary()

