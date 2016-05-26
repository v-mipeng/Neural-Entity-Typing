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

from satori_multi_train import SatoriDataset


def setup_datastream(path, config, word2id = None, word_freq = None):
    dataset = SatoriDataset(path, config.to_label_id, word2id, word_freq)
    it = SequentialScheme(dataset.num_examples, config.batch_size)
    stream = DataStream(dataset, iteration_scheme=it)
    # Add mask
    stream = Padding(stream, mask_sources=['context'], mask_dtype='int32')
    # Debug
    return dataset, stream
