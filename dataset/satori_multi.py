# -*- coding : utf-8 -*-
import logging
import random
import numpy

import cPickle

from picklable_itertools import iter_

from fuel.datasets import Dataset, IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import IterationScheme, ConstantScheme, IndexScheme, ShuffledExampleScheme
from fuel.transformers import Batch, Mapping, SortMapping, Unpack, Padding, Transformer

import sys
import os
import re
import codecs
from collections import OrderedDict

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

regex1 = re.compile("[A-Z]+")
regex2 = re.compile("[a-z]+")
regex3 = re.compile("[\d,]+$")
regex4 = re.compile("[^\w/,]")
def stem(word):
    word = word.decode('utf-8').encode('ASCII')
    if len(regex4.findall(word)) > 0:
        return "<SYM>"      # Add mask on words like http://sldkf, 20,409,300
    word = regex1.sub("AA", word)
    word = regex2.sub("aa", word)
    word = regex3.sub("00", word)
    return word.encode('utf-8')

class SatoriDataset(IndexableDataset):
    def __init__(self, path, to_label_id, word2id, word_freq = None, **kwargs):
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

        if word2id is None:
            self.extract_vocab = True
            self.word2id = {}
            self.word2id['<UNK>'] = len(self.word2id)
            self.word2id['<BOS>'] = len(self.word2id)
        else:
            self.extract_vocab = False
            self.word2id = word2id
        if word_freq is None:
            self.word_freq = self.count_word_freq()
        else:
            self.word_freq = word_freq
        self.path = path
        self.to_label_id = to_label_id
        self._context = []
        self._mention_begin = []
        self._mention_end = []
        self._label = []
        self.load_data()
        self.vocab_size = len(self.word2id)
        super(SatoriDataset, self).__init__(
            indexables = OrderedDict([('context', self._context), ('mention_begin', self._mention_begin), ('mention_end', self._mention_end), ('label', self._label)]),
                                     **kwargs)


    def count_word_freq(self):
        '''
        Count word frequency.
        '''
        files = listdir(self.path)
        self.word_freq = dict()
        for file in files:
            if os.path.isfile(os.path.join(basedir,file)):
                with codecs.open(os.path.join(self.path,file),"r", "UTF-8") as f:
                    for line in f:
                        try:
                            line = line.strip()
                            array = line.split('\t')
                            mention = array[0]
                            type = array[1]
                            context = array[2]
                            contexts = context.split(' ')
                            for word in contexts:
                                word = word.decode('utf-8').lower()
                                if word in word_freq:
                                    self.word_freq[word] += 1
                                else:
                                    self.word_freq[word] = 1
                        except Exception as e:
                            print("find error in counting word frequency!")

    def load_data(self):

        def get_mention_index(context, mention):
            length = len(mention)
            for index in (i for i, e in enumerate(context) if e==mention[0]):
                if context[index:index+length]==mention:
                    return index, index+length
            return -1, -1

        contexts = []
        files = os.listdir(self.path)
        for file in files:
            with codecs.open(os.path.join(self.path, file), "r", "UTF-8") as f:
                for line in f:
                    array = line.strip().split("\t")
                    mention = array[0].split(" ")                                 
                    context = array[len(array)-1].split(" ")                                  
                    begin, end = get_mention_index(context,mention)
                    # Add mask on mention and context
                    for i in range(len(context)):
                        word = context[i].decode('utf-8').lower()
                        if (word not in word_freq) or (word_freq[word] < 10):
                            context[i] = stem(context[i])
                    for i in range(len(mention)):
                        word = mention[i].decode('utf-8').lower()
                        if (word not in word_freq) or (word_freq[word] < 10):
                            mention[i] = stem(mention[i])
                    # Extract word2id table
                    if self.extract_vocab:
                        for word in context:
                            if word not in self.word2id:
                                self.word2id[word] = len(self.word2id)
                    if begin < 0:
                        continue
                    if array[1] in self.to_label_id:
                        self._label += [numpy.int32(self.to_label_id[array[1]])]
                    else:
                        continue
                    if begin == 0:
                        contexts += ['<BOS>',context]
                    self._mention_begin += [numpy.int32(begin)]
                    self._mention_end += [numpy.int32(end)]
        dataset = ()
        for context in contexts:
            self._context += [self.to_word_ids(context)]

    def to_word_id(self, w):
        if w in self.word2id:
            return self.word2id[w]
        else:
            return self.word2id['<UNK>']

    def to_word_ids(self, context):
        return numpy.array([self.to_word_id(x) for x in context], dtype=numpy.int32)


class SatoriShuffleScheme(IterationScheme):
    requests_examples = True
    def __init__(self, dataset_size, *args, **kwargs):
        self.dataset_size = dataset_size
        super(SatoriShuffleScheme, self).__init__(*args, **kwargs)

    def get_request_iterator(self):
        indexes = numpy.arange(self.dataset_size)
        numpy.random.shuffle(indexes)
        return iter_(indexes)

# -------------- DATASTREAM SETUP --------------------


class _balanced_batch_helper(object):
    def __init__(self, key):
        self.key = key
    def __call__(self, data):
        return data[self.key].shape[0]  # sort key

def setup_datastream(path, config, word2id = None):
    dataset = SatoriDataset(path, config.to_label_id, word2id)
    it = ShuffledExampleScheme(dataset.num_examples)
    stream = DataStream(dataset, iteration_scheme=it)
       
    # Sort sets of multiple batches to make batches of similar sizes
    stream = Batch(stream, iteration_scheme=ConstantScheme(config.batch_size * config.sort_batch_count))
    comparison = _balanced_batch_helper(stream.sources.index('context'))
    stream = Mapping(stream, SortMapping(comparison))
    stream = Unpack(stream)

    # Add mask
    stream = Batch(stream, iteration_scheme=ConstantScheme(config.batch_size))
    stream = Padding(stream, mask_sources=['context'], mask_dtype='int32')
    # Debug
    for data in stream.get_epoch_iterator():
        d = data
        pass    
    return dataset, stream
