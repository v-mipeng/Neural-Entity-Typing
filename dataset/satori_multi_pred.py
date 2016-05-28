# -*- coding : utf-8 -*-
import logging
import random
import numpy
import glob

import cPickle

from picklable_itertools import iter_

from fuel.datasets import Dataset, IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import IterationScheme, ConstantScheme, IndexScheme, ShuffledExampleScheme, SequentialScheme
from fuel.transformers import Batch, Mapping, SortMapping, Unpack, Padding, Transformer

import sys
import os
import re
import codecs
from collections import OrderedDict

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

regex1 = re.compile(r"[^\w/,]")
regex2 = re.compile(r"^[\d,]+$")
regex3 = re.compile(r"[A-Z]+")
regex4 = re.compile(r"[a-z]+")
regex5 = re.compile(r"\d+")


def stem(word):
    if len(regex1.findall(word)) > 0:
        return "<SYM>"      # Add mask on words like http://sldkf, 20,409,300
    if regex2.match(word):
        return "00"
    word = regex3.sub("AA", word)
    word = regex4.sub("aa", word)
    word = regex5.sub("00", word)
    return word.encode('utf-8')

class SatoriDataset(IndexableDataset):
    def __init__(self, samples , to_label_id, word2id, word_freq = None, **kwargs):
        '''
        Construct database.
        File format should be : mention TAB type TAB  context
        @param path: directory path of source files, this program will load all the files under this directory as data
        
        @param to_label_id: label2id dictionary

        @param word2id: dictonary mapping word to integer value. For training dataset, this is optional while for test dataset
                        you must provide it extracted from training dataset.
        
        @param word_freq: dictonary recoding frequency of words. The program will add mask to sparse words with this information
                          For training dataset, this is optional while for test dataset you must provide it extracted from 
                          training dataset.
        '''
        assert samples is not None
        assert to_label_id is not None
        assert word2id is not None
        assert word_freq is not None

        self.word2id = word2id
        self.word_freq = word_freq
        self.samples = samples
        self.to_label_id = to_label_id
        self.context = []
        self.mention = []
        self._context = []
        self._mention_begin = []
        self._mention_end = []
        self.label = [None]*len(samples)
        self.prepare_data()
        self.vocab_size = len(self.word2id)
        super(SatoriDataset, self).__init__(
            indexables = OrderedDict([('context', self._context), ('mention_begin', self._mention_begin), ('mention_end', self._mention_end)])
            ,**kwargs)



    def prepare_data(self):

        def get_mention_index(context, mention):
            length = len(mention)
            for index in (i for i, e in enumerate(context) if e==mention[0]):
                if context[index:index+length]==mention:
                    return index, index+length
            return -1, -1

        contexts = []
        offset = 0
        for mention, context in self.samples:
            try:
                mention_tokens = mention.split(" ")                                 
                context_tokens = context.split(" ")                                  
                begin, end = get_mention_index(context_tokens, mention_tokens)

                if begin < 0:
                    raise Exception("Cannot find mention: %s in given context: %s. Skip the sample!" %(mention,context))
                # Add mask on mention and context
                for i in range(len(context_tokens)):
                    word = context_tokens[i].decode('utf-8').lower()
                    if (word not in self.word_freq) or (self.word_freq[word] < 10):
                        context_tokens[i] = stem(context_tokens[i])
                for i in range(len(mention_tokens)):
                    word = mention_tokens[i].decode('utf-8').lower()
                    if (word not in self.word_freq) or (self.word_freq[word] < 10):
                        mention_tokens[i] = stem(mention_tokens[i])

                # Add begin_of_sentence label
                contexts += [['<BOS>']+context_tokens]   
                self._mention_begin += [numpy.int32(begin)]
                self._mention_end += [numpy.int32(end)]
                self.mention += [mention]
                self.context += [context]
            except:
                self.label[offset] = "UNKNOWN"
            offset += 1

        for context in contexts:
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
    dataset = SatoriDataset(samples, config.to_label_id, word2id, word_freq)
    it = SequentialScheme(dataset.num_examples, config.batch_size)
    stream = DataStream(dataset, iteration_scheme=it)
    # Add mask
    stream = Padding(stream, mask_sources=['context'], mask_dtype='int32')
    # Debug
    return dataset, stream
