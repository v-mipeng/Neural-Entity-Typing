# -*- coding : utf-8 -*-
import logging
import random
import numpy
import theano

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

import codecs
from abc import abstractmethod, ABCMeta

from base import *
from base import _balanced_batch_helper
from error import *
from resource.dbpedia import DBpedia
from error import *
import time
from __builtin__ import super

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

from multi_time_lstm import MTL

class TDLD(MTL):
    '''
    Triple direction lstm dataset
    '''
    def __init__(self, config):
        super(TDLD, self).__init__(config)
        self.provide_souces = ('to_begin', 'to_end','label')
        self.label_index = 3
        self.need_mask_sources = {'to_begin':self.config.int_type, 'to_end': self.config.int_type}    
        self.assit_words = ('<UNK>','<BOS>','<END>')
        self.compare_source = 'to_begin'

    def parse_one_sample(self, line, with_label = True):
        '''
        Parse one sample
        '''
        if self.word_freq is None:
            if self.train_data_path is None:
                raise Exception("word_freq cannot be None!")
            else:
                self.get_word_freq(self.train_data_path)
                self.save_word_freq()
        if self.word2id is None:
            if self.train_data_path is None:
                raise Exception("word2id cannot be None!")
            else:
                self.get_word2id(self.train_data_path)
                self.save_word2id()
        if self.config.develop: # Assuming that during developing, all the data has been pre-processed                
            array = line.split('\t')
            mention_tokens = array[0].split(' ')
            context_tokens = array[len(array)-1].split(' ')
        else:
            line = split_hyphen(line)
            array = line.split("\t")
            mention = array[0]
            context = array[len(array)-1]
            char_begin = context.find(mention)
            if char_begin == -1:
                raise MentionNotFoundError()
            context = filter_context(char_begin, char_begin+len(mention), context)
            mention_tokens = tokenize(mention)                                
            context_tokens = tokenize(context)
        begin, end = get_mention_index(context_tokens,mention_tokens)
        if begin < 0:
            raise MentionNotFoundError()
        if with_label:
            if array[1] in self.config.to_label_id:
                _label = numpy.int32(self.config.to_label_id[array[1]]).astype(self.config.int_type)
            else:
                raise FileFormatError("Label%s not defined!" % array[1])
        # Add mask on mention and context_tokens
        for i in range(len(context_tokens)):
            word = context_tokens[i].decode('utf-8').lower()
            if i >= begin and i < end:
                if (word not in self.word_freq) or (self.word_freq[word] < self.config.sparse_mention_threshold):
                    context_tokens[i] = self.stem(context_tokens[i])                         
            elif (word not in self.word_freq) or (self.word_freq[word] < self.config.sparse_word_threshold):
                context_tokens[i] = self.stem(context_tokens[i])                         
        for i in range(len(mention_tokens)):
            word = mention_tokens[i].decode('utf-8').lower()
            if (word not in self.word_freq) or (self.word_freq[word] < self.config.sparse_mention_threshold):
                mention_tokens[i] = self.stem(mention_tokens[i])
        # Add begin_of_sentence label
        _to_begin = self.to_word_ids(['<BOS>']+[context_tokens[end-1-i] for i in range(end)])
        _to_end = self.to_word_ids([context_tokens[i] for i in range(begin, len(context_tokens))]+['<END>'])
        mention = " ".join(mention_tokens)
        context = " ".join(context_tokens)
        if with_label:
            return (_to_begin, _to_end, _label, mention, context)
        else:
            return (_to_begin, _to_end, mention, context)    