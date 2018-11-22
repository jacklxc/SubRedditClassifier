from __future__ import print_function, division

from embedding_reader import EmbeddingReader
from spreadsheet_classifier import *
import argparse
from tqdm import tqdm

dataFile = "data/contents.20.txt"
labelFile = "data/labels.20.txt"
repFile = "glove.6B.50d.txt"
model_name = "SuperSimpleLSTMClassifier"
testSize = 100
randomizeTestSet = True
MAX_NUM_WORDS = 400000

reset_random_seed(0)

ER = EmbeddingReader(repFile,MAX_NUM_WORDS)
vacab = set(ER.embeddings_index.keys())
sd = SpreadsheetData(dataFile, labelFile, testSize, vacab, randomizeTestSet,MAX_NUM_WORDS = MAX_NUM_WORDS)

embedding_matrix = ER.make_embedding_matrix(sd.word_index)
sce = SpreadsheetClassificationExecution(sd, embedding_matrix, model_name)
