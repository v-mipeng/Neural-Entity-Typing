#!/usr/bin/env python
import sys
import logging
import numpy
import os
import importlib
import codecs

import theano
import nltk

from nltk.tokenize.punkt import PunktSentenceTokenizer
from blocks.graph import ComputationGraph
from blocks.model import Model


try:
    from blocks_extras.extensions.plot import Plot
    plot_avail = True
except ImportError:
    plot_avail = False
    print "No plotting extension available."


from dataset import multi_pred
from paramsaveload import SaveLoadParams

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

sys.setrecursionlimit(2000000)

nltk.data.path.append(r"D:\Data\NLTK Data")

model_name = "multi_time_lstm"
config = importlib.import_module('.%s' % model_name, 'config')

# load word2id and word_freq
if os.path.exists(config.word2id_path):
    word2id = {}
    with codecs.open(config.word2id_path, "r", encoding = "UTF-8") as f:
        for line in f:
            array = line.split('\t')
            word2id[array[0]] = int(array[1])
else:
    raise("Cannot find vocabulary file!")
if os.path.exists(config.word_freq_path):
    word_freq = {}
    with codecs.open(config.word_freq_path, "r", encoding = "UTF-8") as f:
        for line in f:
            array = line.split('\t')
            word_freq[array[0]] = int(array[1])
else:
    raise("Cannot find word frequency file!")

model_path = os.path.join(config.model_path, model_name+"_gpu.pkl")
    
# Build model
m = config.Model(config, len(word2id))
cg = ComputationGraph(m.sgd_cost)

# Initialize
model = Model(m.sgd_cost)   
initializer = SaveLoadParams(model_path,model)
initializer.do_load()
    
# Build predictor
cg = ComputationGraph(m.pred)
f_pred = theano.function(cg.inputs,m.pred)
pred_inputs = cg.inputs

# Do prediction and write the result to file
label2id = config.to_label_id
id2label = {}
for item in label2id.items():
    id2label[item[1]] = item[0]


def predict(char_begins, char_ends, contexts):
    '''
    Predict type of mentions.

    @param char_begins: a list of integers with each value represents the start index of a mention within corresponding context

    @param char_ends: a lsit of integer with each value represents the end index of a mention within corresponding context

    @param contexts: a list of strings

    @return a list string 
    '''
    ds, stream = multi_pred.setup_datastream(char_begins, char_ends, contexts, word2id, word_freq)
    labels = ds.label
    offset = 0
    for inputs in stream.get_epoch_iterator():
        input_len = len(inputs[stream.sources.index(pred_inputs[0].name)])

        label_ids = f_pred(inputs[stream.sources.index(pred_inputs[0].name)],
                            inputs[stream.sources.index(pred_inputs[1].name)],
                            inputs[stream.sources.index(pred_inputs[2].name)])
        for label_id in label_ids:
            while labels[offset] == "UNKNOWN":
                offset += 1
            labels[offset] = id2label[label_id]
            offset += 1
    return labels

def predict(mentions, contexts):
    char_begins = []
    for mention,context in zip(mentions,contexts):
        begin = context.find(mention)
        end = begin + len(begin)
        char_begins += [begin]
        char_ends += [end]
    predict(char_begins,char_ends,contexts)

