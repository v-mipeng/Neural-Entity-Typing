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
from error import *
import time
from __builtin__ import super

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

class BasicDataset(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_train_stream(self, data_path, valid_portion = 0.0):
        raise NotImplementedError('subclasses must override get_train_stream()!')

    @abstractmethod
    def get_test_stream(self, datapath):
        raise NotImplementedError('subclasses must override get_test_stream()!')

    @abstractmethod
    def get_predict_stream(self, datapath):
        raise NotImplementedError('subclasses must override get_predict_stream()!')

class MTL(BasicDataset):
    '''
    Base Multiple_Time_LSTM dataset.
    '''
    def __init__(self, config):
        '''
        @param config: Model config module
        '''
        self.config = config
        self.word2id = None
        self.word_freq = None
        self.provide_souces = ('context', 'mention_begin', 'mention_end','label')
        self.label_index = 3
        self.need_mask_sources = {'context':self.config.int_type}
        self.compare_source = 'context'
        # Regular expression for stemming
        self.regex1 = re.compile(r"[^\w/,]")
        self.regex2 = re.compile(r"^[\d,]+$")
        self.regex3 = re.compile(r"[A-Z]+")
        self.regex4 = re.compile(r"[a-z]+")
        self.regex5 = re.compile(r"\d+")
        self.assit_words = ('<UNK>','<BOS>')
        self.train_data_path = None

        self.init()

    def init(self):

        if os.path.exists(self.config.word2id_path):
            self.load_word2id(self.config.word2id_path)

        if os.path.exists(self.config.word_freq_path):
            self.load_word_freq(self.config.word_freq_path)

    def get_train_stream(self, data_path, valid_portion = 0.0):
        '''
        Load dataset from given data_path, and split it into training dataset and validation dataset.
        Validation dataset size = total_dataset_size*valid_protion

        @param data_path: Path of a file or a folder. If it is a file path, extract dataset from this file, 
                          otherwise extract dataset from all the files with ".txt" surfix under given folder.
        @param valid_portion: a float value ~[0,1), if it is 0 no validation set will be returned

        @return: if valid_portion ~(0,1), return training_datastream, valid_datastream
                 with training_datastream is shuffled and valid_datastream is sequencial,
                 else return training_datastream
        '''
        self.train_data_path = data_path
        dataset = self.load_dataset(data_path)
        assert valid_portion < 1
        if valid_portion > 0:
            #Split dataset into training set and validation dataset
            train_data, valid_data = split_train_valid(dataset, valid_portion)
            train_ds = self.construct_dataset(train_data)
            valid_ds = self.construct_dataset(valid_data)
            train_stream = self.construct_shuffled_stream(train_ds)
            valid_stream = self.construct_sequencial_stream(valid_ds)
            return train_stream, valid_stream
        else:
            train_ds = self.construct_dataset(dataset)
            train_stream = self.construct_shuffled_stream(train_ds)
            return train_stream

    def get_test_stream(self, data_path):
        '''
        Load test dataset from given data_path

        @param data_path: Path of a file or a folder. If it is a file path, extract dataset from this file, 
                          otherwise extract dataset from all the files with ".txt" surfix under given folder.
        @return: Return test_datastream, test_dataset

        '''
        dataset = self.load_dataset(data_path)
        test_ds = self.construct_dataset(dataset)
        test_stream = self.construct_sequencial_stream(test_ds)
        return test_stream, dataset

    def get_predict_stream(self, data_path):
        '''
        Load predict dataset from given data_path

        @param data_path: Path of a file or a folder. If it is a file path, extract dataset from this file, 
                          otherwise extract dataset from all the files with ".txt" surfix under given folder.
        @return: Return predict_datastream, predict_dataset

        '''
        dataset = self.load_dataset(data_path, with_label = False)
        predict_ds = self.construct_dataset(dataset, False)
        predict_stream = self.construct_sequencial_stream(predict_ds)
        return predict_stream, dataset

    def construct_dataset(self, dataset, with_label = True):
        dataset = zip(*dataset)
        pairs = []
        if with_label:
            for i in range(self.label_index+1):
                pairs.append((self.provide_souces[i], dataset[i]))
            return IndexableDataset(indexables = OrderedDict(pairs))
        else:
            for i in range(self.label_index):
                pairs.append((self.provide_souces[i], dataset[i]))
            return IndexableDataset(indexables = OrderedDict(pairs))

    def construct_shuffled_stream(self, dataset):
        it = ShuffledExampleScheme(dataset.num_examples)
        stream = DataStream(dataset, iteration_scheme=it)
        # Sort sets of multiple batches to make batches of similar sizes
        stream = Batch(stream, iteration_scheme=ConstantScheme(self.config.batch_size * self.config.sort_batch_count))
        comparison = _balanced_batch_helper(stream.sources.index(self.compare_source))
        stream = Mapping(stream, SortMapping(comparison))
        stream = Unpack(stream)
        stream = Batch(stream, iteration_scheme=ConstantScheme(self.config.batch_size))
        # Add mask
        for source in self.need_mask_sources.iteritems():
            stream = Padding(stream, mask_sources=[source[0]], mask_dtype= source[1])
        return stream

    def construct_sequencial_stream(self, dataset):
        it = SequentialScheme(dataset.num_examples, self.config.batch_size)
        stream = DataStream(dataset, iteration_scheme=it)
        # Add mask
        for source in self.need_mask_sources.iteritems():
            stream = Padding(stream, mask_sources=[source[0]], mask_dtype= source[1])
        return stream

    def load_dataset(self, data_path, with_label = True):
        print("Load dataset from %s..." %os.path.abspath(data_path))
        dataset = []
        if os.path.isdir(data_path):
            files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
        else:
            data_path, file = os.path.split(data_path)
            files = [file]
        for file in files:
            with codecs.open(os.path.join(data_path, file), "r", "UTF-8", errors = "ignore") as f:
                for line in f:
                    try:
                        dataset.append(self.parse_one_sample(line.strip(), with_label))
                    except MentionNotFoundError:
                        continue
                    except Exception as e:
                        try:
                            print(e.message)
                            print("Find Error during loading dataset!")
                        except:
                            print("Find Error during loading dataset!")
        print("Done!")
        return dataset

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
        _context = self.to_word_ids(['<BOS>']+context_tokens)
        _mention_begin = numpy.int32(begin).astype(self.config.int_type)
        _mention_end = numpy.int32(end).astype(self.config.int_type)
        mention = " ".join(mention_tokens)
        context = " ".join(context_tokens)
        if with_label:
            return (_context, _mention_begin, _mention_end, _label, mention, context)
        else:
            return (_context, _mention_begin, _mention_end, mention, context)

    def get_word2id(self, data_path):
        '''
        Construct word2id table.
        '''
        if self.word_freq is None:
            self.get_word_freq(data_path)
        if os.path.isdir(data_path):
            files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
        else:
            data_path, file = os.path.split(data_path)
            files = [file]
        self.word2id = dict()
        for word in self.assit_words:
            self.word2id[word] = len(self.word2id)
        for file in files:
            with codecs.open(os.path.join(data_path, file),"r", "UTF-8", errors = "ignore") as f:
                for line in f:
                    try:
                        line = line.strip()
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
                            continue
                        for i in range(len(context_tokens)):
                            word = context_tokens[i].decode('utf-8').lower()
                            if i >= begin and i < end:
                                if (word not in self.word_freq) or (self.word_freq[word] < self.config.sparse_mention_threshold):
                                    context_tokens[i] = self.stem(context_tokens[i])                         
                            elif (word not in self.word_freq) or (self.word_freq[word] < self.config.sparse_word_threshold):
                                    context_tokens[i] = self.stem(context_tokens[i])    
                        for word in context_tokens:
                            if word not in self.word2id:
                                self.word2id[word] = len(self.word2id)
                    except Exception as e:
                        try:
                            print(e.message)
                            print("Find error during construct word2id table!")
                        except:
                            print("Find error during construct word2id table!")

    def get_word_freq(self, data_path):
        '''
        Count word frequency.
        '''
        if os.path.isdir(data_path):
            files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
        else:
            data_path, file = os.path.split(data_path)
            files = [file]
        self.word_freq = dict()
        for file in files:
            with codecs.open(os.path.join(data_path, file),"r", "UTF-8", errors = "ignore") as f:
                for line in f:
                    try:
                        line = line.strip()
                        if self.config.develop: # Assuming that during developing, all the data has been pre-processed                
                            context_tokens = self.get_context(line.strip()).split(' ')
                        else:
                            line = split_hyphen(line)
                            context = self.get_context(line)
                            context_tokens = tokenize(context)

                        for word in context_tokens:
                            word = word.decode('utf-8').lower()
                            if word in self.word_freq:
                                self.word_freq[word] += 1
                            else:
                                self.word_freq[word] = 1
                    except Exception as e:
                        try:
                            print(e.message)
                            print("Find error during counting word frequency!")
                        except:
                            print("Find error during counting word frequency!")

    def get_context(self, line):
        array =  line.split('\t')
        return array[len(array)-1]

    def stem(self, word):
        if len(self.regex1.findall(word)) > 0:
            return "<SYM>"      # Add mask on words like http://sldkf, 20,409,300
        if self.regex2.match(word):
            return "00"
        word = self.regex3.sub("AA", word)
        word = self.regex4.sub("aa", word)
        word = self.regex5.sub("00", word)
        return word.encode('utf-8')

    def to_word_id(self, w):
        if w in self.word2id:
            return self.word2id[w]
        else:
            return self.word2id['<UNK>']

    # This part can be deleted later.
    def to_word_ids(self, tokens):
        return numpy.array([self.to_word_id(x) for x in tokens], dtype= self.config.int_type)

    def load_word2id(self, path):
        self.word2id = load_dic(self.config.word2id_path)

    def load_word_freq(self, path):
        self.word_freq = load_dic(self.config.word_freq_path)

    def save_word2id(self):
        save_dic(self.config.word2id_path, self.word2id)  

    def save_word_freq(self):
        save_dic(self.config.word_freq_path, self.word_freq)

class MTLD(MTL):

    '''
    Multiple_Time_LSTM_with_Dbpedia dataset.
    '''
    from resource.dbpedia import DBpedia
    def __init__(self,config):
        '''
        @param config: Model config module
        '''
        super(MTLD, self).__init__(config)
        self.dbpedia = DBpedia()
        self.type2id = None
        if os.path.exists(config.type2id_path):
            self.load_type2id()
        else:
            raise Exception("Type2id cannot be None!")
        self.provide_souces = ('context', 'mention_begin', 'mention_end', 'type', 'type_weight', 'label')
        self.need_mask_sources = {'context':self.config.int_type, 'type':self.config.int_type, 'type_weight': theano.config.floatX}
        self.label_index = 5

    def parse_one_sample(self, line, with_label = True):
        # Extract mention matched types
        sample = super(MTLD, self).parse_one_sample(line, with_label)
        mention = sample[-2]
        pairs = self.dbpedia.get_match_entities(mention)
        if pairs is not None:
            types, indegrees = zip(*pairs)
            _type = numpy.array([self.type2id[type] for type in types], self.config.int_type)
            weights = [numpy.sqrt(indegree) for indegree in indegrees]  # log indegree
            s = sum(weights) # max normalization
            _type_weight = numpy.array([weight/s for weight in weights], dtype = theano.config.floatX)
        else:
            _type = numpy.array([self.type2id["UNKNOWN"]],dtype= self.config.int_type)
            _type_weight = numpy.array([1.0], dtype= theano.config.floatX)
        return sample[0:3]+ (_type, _type_weight) + sample[3:]

    def load_type2id(self):
        self.type2id = load_dic(self.config.type2id_path)

class WLSTMD(MTL):
    '''
    Weighted lstm dataset.
    '''
    def __init__(self,config):
        '''
        @param config: Model config module
        '''
        super(WLSTMD, self).__init__(config)
        self.provide_souces = ('context', 'distance', 'label')
        self.need_mask_sources = {'context':self.config.int_type, 'distance':self.config.int_type}
        self.label_index = 2

    def parse_one_sample(self, line, with_label = True):
        # Extract mention matched types
        sample = super(WLSTMD, self).parse_one_sample(line, with_label)
        mention_begin = sample[1]
        mention_end = sample[2]
        _distance = numpy.asarray([mention_begin+1-i for i in range(mention_begin+1)]+\
            [0 for i in range(mention_begin+1, mention_end+1)]+\
            [i-mention_end for i in range(mention_end+1, len(sample[0]))], dtype = self.config.int_type)
        if with_label:
            return (sample[0], _distance, sample[3])
        else:
            return (sample[0], _distance)