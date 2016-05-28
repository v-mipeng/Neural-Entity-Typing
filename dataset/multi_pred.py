'''
Dataset for prediction
'''

# -*- coding : utf-8 -*-
import logging
import random
import numpy
import glob
import cPickle
from picklable_itertools import iter_
import sys
import os
import re
import codecs
import nltk

from fuel.datasets import Dataset, IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import IterationScheme, ConstantScheme, IndexScheme, ShuffledExampleScheme, SequentialScheme
from fuel.transformers import Batch, Mapping, SortMapping, Unpack, Padding, Transformer
from nltk.tokenize.punkt import PunktSentenceTokenizer
from collections import OrderedDict

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

regex1 = re.compile(r"[^\w/,]")
regex2 = re.compile(r"^[\d,]+$")
regex3 = re.compile(r"[A-Z]+")
regex4 = re.compile(r"[a-z]+")
regex5 = re.compile(r"\d+")

nltk.data.path.append(r"D:\Data\NLTK Data")


def stem(word):
    if len(regex1.findall(word)) > 0:
        return "<SYM>"      # Add mask on words like http://sldkf, 20,409,300
    if regex2.match(word):
        return "00"
    word = regex3.sub("AA", word)
    word = regex4.sub("aa", word)
    word = regex5.sub("00", word)
    return word.encode('utf-8')

class PredDataset(IndexableDataset):
    def __init__(self, char_begins, char_ends, contexts, word2id, word_freq, **kwargs):
        '''
        Construct database for prediction
        @param char_begins: a list of character begins of mentions within contexts

        @param char_ends: a list of character ends of mentions within contexts. context[char_begin:char_end] = mention
        

        @param word2id: dictonary mapping word to integer value. For training dataset, this is optional while for test dataset
                        you must provide it extracted from training dataset.
        
        @param word_freq: dictonary recoding frequency of words. The program will add mask to sparse words with this information
                          For training dataset, this is optional while for test dataset you must provide it extracted from 
                          training dataset.
        '''

        assert path is not None
        assert to_label_id is not None
        assert word2id is not None
        assert word_freq is not None

        self.word2id = word2id
        self.word_freq = word_freq
        self.path = path
        self.to_label_id = to_label_id
        self.context = []
        self.mention = []
        self._context = []
        self._mention_begin = []
        self._mention_end = []
        self._label = []
        self.load_data()
        self.vocab_size = len(self.word2id)
        super(SatoriDataset, self).__init__(
            indexables = OrderedDict([('context', self._context), ('mention_begin', self._mention_begin), ('mention_end', self._mention_end), ('label', self._label)]),
                                     **kwargs)

    def construct_dataset(self, char_begins, char_ends, contexts):

        def get_mention_index(context, mention):
            length = len(mention)
            for index in (i for i, e in enumerate(context) if e==mention[0]):
                if context[index:index+length]==mention:
                    return index, index+length
            return -1, -1

        offset = 0
        for char_begin, char_end, context in zip(char_begins, char_ends, contexts):
            try:
                mention = context[char_start,char_end]
                mention_tokens = nltk.word_tokenize(mention)
                context_tokens = None
                for start, end in PunktSentenceTokenizer.span_tokenize(context):
                    if start <= char_start and end >= char_end:
                        contexts = nltk.word_tokenize(context[start,end])
                        break
                if context_tokens is None:
                    raise Exception("Cannot find mention: %s in given context: %s" %(mention,context))
                begin, end = get_mention_index(context_tokens,mention_tokens)
                for i in range(len(mention_tokens)):
                    if mention_tokens[i] not in self.word_freq or self.word_freq[mention_tokens[i]]<10:
                        mention_tokens[i] = stem(mention_tokens[i])
                for i in range(len(context_tokens)):
                    if context_tokens[i] not in self.word_freq or self.word_freq[context_tokens[i]]<10:
                        context_tokens[i] = stem(context_tokens[i])
                self.mention += [mention_tokens]
                self.context += [['<BOS>']+context_tokens]
                self._mention_begin += [numpy.int32(begin)]
                self._mention_end += [numpy.int32(end)]
                offset += 1
            except Exception as e:
                self.label[offset] = "UNKNOWN"
                try:
                    print(e.message)
                    print("Find Error during loading dataset!")
                except:
                    print("Find Error during loading dataset!")
            # Add begin_of_sentence label
        for context in self.context:
            self._context += [self.to_word_ids(context)]

    def to_word_id(self, w):
        if w in self.word2id:
            return self.word2id[w]
        else:
            return self.word2id['<UNK>']

    def to_word_ids(self, context):
        return numpy.array([self.to_word_id(x) for x in context], dtype=numpy.int32)

# -------------- DATASTREAM SETUP --------------------

def setup_datastream(samples, config, word2id, word_freq):
    dataset = PredDataset(samples, word2id, word_freq)
    it = SequentialScheme(dataset.num_examples, config.batch_size)
    stream = DataStream(dataset, iteration_scheme=it)
    # Add mask
    stream = Padding(stream, mask_sources=['context'], mask_dtype='int32') 
    return dataset, stream
