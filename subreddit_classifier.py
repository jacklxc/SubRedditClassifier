from __future__ import print_function, division

from embedding_reader import EmbeddingReader
from spreadsheet_classifier import *
import argparse
from tqdm import tqdm
import pickle
import numpy as np
print("Complex label!")
dataFile = "data/contents.100.txt"
labelFile = "data/labels.100.txt"
repFile = "/nas/evidx/corpora/XiangciLi/embeddings/glove.6B.300d.txt"
model_name = "CNNBiLSTM"
randomizeTestSet = True
MAX_NUM_WORDS = 400000
limit = 100000
testSize = int(limit*0.1)

reset_random_seed(0)

ER = EmbeddingReader(repFile,MAX_NUM_WORDS)
vacab = set(ER.embeddings_index.keys())
sd = SpreadsheetData(dataFile, labelFile, testSize, vacab, limit, randomizeTestSet,MAX_NUM_WORDS = MAX_NUM_WORDS)

embedding_matrix = ER.make_embedding_matrix(sd.word_index)
sce = SpreadsheetClassificationExecution(sd, embedding_matrix, model_name)
pickle.dump(np.argmax(sce.prediction,axis=1),open("prediction_complex.pkl","wb"))
pickle.dump(np.argmax(sd.y_test,axis=1),open("true_label_complex.pkl","wb"))
pickle.dump(sd.labels,open("label_name_complex.pkl","wb"))