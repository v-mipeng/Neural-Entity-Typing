# -*- coding : utf-8 -*-
import logging
import numpy
import re
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from config.lstm_config import BasicConfig as config
import ntpath
import os

nltk.data.path.append(config.nltk_data_path)

import codecs
from abc import abstractmethod, ABCMeta

from error import MentionNotFoundError

tokenizer = PunktSentenceTokenizer()
regex1 = re.compile(r"(^|\s)([a-zA-Z]+)/([a-zA-Z]+)(?=$|\s)")
regex2 = re.compile(r"(\w+)-(?=\w+)")

class _balanced_batch_helper(object):
    def __init__(self, key):
        self.key = key
    def __call__(self, data):
        return data[self.key].shape[0]

def split_train_valid(data , valid_portion):
    '''
    Split dataset into training set and validation dataset
    '''
    idxes = range(len(data))
    numpy.random.shuffle(idxes)
    train_data = []
    valid_data = []
    for idx in range(int(numpy.floor(valid_portion*len(data)))):
        valid_data.append(data[idxes[idx]])
    for idx in range(int(numpy.floor(valid_portion*len(data))),len(data)):
        train_data.append(data[idxes[idx]])
    return train_data, valid_data

def load_dic(path):
    dic = {}
    with codecs.open(path, "r", encoding = "UTF-8") as f:
        for line in f:
            array = line.split('\t')
            dic[array[0]] = int(array[1])
    return dic

def save_dic(path, dictionary):
    dir = ntpath.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with codecs.open(path, "w+", encoding = "UTF-8") as f:
        for key, value in dictionary.iteritems():
            f.write("%s\t%s\n" % (key, value))

def get_mention_index(context_tokens, mention_tokens):
    length = len(mention_tokens)
    for index in (i for i, e in enumerate(context_tokens) if e==mention_tokens[0]):
        if context_tokens[index:index+length]==mention_tokens:
            return index, index+length
    return -1, -1

def filter_context(char_begin, char_end, context):
    idxs = tokenizer.span_tokenize(context)
    contexts = []
    for begin, end in idxs:
        contexts.append(context[begin:end])
    count = 0
    for begin, end in idxs:
        if begin <= char_begin and end >= char_end:
            context = contexts[count]
            break
        count += 1
    if count == len(contexts):
        raise MentionNotFoundError()
    else:
        return context

def tokenize(input_string):
    '''
    Tokenize input string into tokens

    @return a list of tokens
    '''
    return nltk.word_tokenize(input_string)

def is_out_encode(line):
    try:
        line.decode('utf-8').encode('ascii')
        return True
    except:
        False

def split_hyphen(line):
    line = regex1.sub("\g<1>\g<2> / \g<3>", line)
    line = regex2.sub("\g<1> - ", line)
    return line

