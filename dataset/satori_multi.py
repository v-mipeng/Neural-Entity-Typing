# -*- coding : utf-8 -*-
import logging
import random
import numpy

import cPickle

from picklable_itertools import iter_

from fuel.datasets import Dataset, IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import IterationScheme, ConstantScheme, IndexScheme, ShuffledExampleScheme, SequentialScheme
from fuel.transformers import Batch, Mapping, SortMapping, Unpack, Padding, Transformer

import sys
import os
import codecs
from collections import OrderedDict

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

class SatoriDataset(IndexableDataset):
    def __init__(self, path, to_label_id, word2id, **kwargs):
        if word2id is None:
            self.extract_vocab = True
            self.word2id = {}
            self.word2id['<UNK>'] = len(self.word2id)
        else:
            self.extract_vocab = False
            self.word2id = word2id
        self.path = path
        self.to_label_id = to_label_id
        self.context = []
        self.mention = []
        self._context = []
        self._mention_end = []
        self._label = []
        self.load_data()
        self.vocab_size = len(self.word2id)
        super(SatoriDataset, self).__init__(
            indexables = OrderedDict([('context', self._context), ('mention_end', self._mention_end), ('label', self._label)]),
                                     **kwargs)

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
                    mention = array[0].split(" ")                                 # get mention
                    context = array[len(array)-1].split(" ")                                  # get context
                    if self.extract_vocab:
                        for word in context:
                            if word not in self.word2id:
                                self.word2id[word] = len(self.word2id)
                    begin, end = get_mention_index(context,mention)
                    if begin < 0:
                        continue
                    if array[1] in self.to_label_id:
                        self._label += [numpy.int32(self.to_label_id[array[1]])]
                    else:
                        continue
                    self.mention += [mention]
                    self._mention_end += [numpy.int32(end - 1)]
                    contexts += [context]
        self.context = contexts
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


class _balanced_batch_helper(object):
    def __init__(self, key):
        self.key = key
    def __call__(self, data):
        return data[self.key].shape[0]  # sort key

def setup_datastream(path, config, word2id = None):
    dataset = SatoriDataset(path, config.to_label_id, word2id)
    it = SequentialScheme(dataset.num_examples, config.batch_size)
    stream = DataStream(dataset, iteration_scheme=it)
    # Add mask
    stream = Padding(stream, mask_sources=['context'], mask_dtype='int32')

    return dataset, stream


# vim: set sts=4 ts=4 sw=4 tw=0 et :
