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

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

class BasicDataset():
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
        if os.path.exists(config.word2id_path):
            self.load_word2id(config.word2id_path)

        if os.path.exists(config.word_freq_path):
            self.load_word_freq(config.word_freq_path)

        self.provide_souces = ('context', 'mention_begin', 'mention_end','label')

        self.regex1 = re.compile(r"[^\w/,]")
        self.regex2 = re.compile(r"^[\d,]+$")
        self.regex3 = re.compile(r"[A-Z]+")
        self.regex4 = re.compile(r"[a-z]+")
        self.regex5 = re.compile(r"\d+")
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
        if self.word_freq is None:
            self.get_word_freq(data_path)
            self.save_word_freq()
        if self.word2id is None:
            self.get_word2id(data_path)
            self.save_word2id()
        t1 = time.time()
        dataset = self.load_dataset(data_path)
        t2 = time.time()
        print("Elapased % seconds for loading dataset" %(t2-t1))
        t1 = time.time()
        assert valid_portion < 1
        if valid_portion > 0:
            #Split dataset into training set and validation dataset
            train_data, valid_data = split_train_valid(dataset, valid_portion)
            t2 = time.time()
            print("Elapased % seconds for split dataset" %(t2-t1))
            t1 = time.time()
            train_ds = self.construct_dataset(train_data)
            valid_ds = self.construct_dataset(valid_data)
            train_stream = self.construct_shuffled_stream(train_ds)
            valid_stream = self.construct_sequencial_stream(valid_ds)
            t2 = time.time()
            print("Elapased % seconds for construct data stream" %(t2-t1))
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
        if self.word_freq is None:
            raise Exception("word_freq cannot be None!")
        if self.word2id is None:
            raise Exception("word2id cannot be None!")
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
        if self.word_freq is None:
            raise Exception("word_freq cannot be None!")
        if self.word2id is None:
            raise Exception("word2id cannot be None!")
        dataset = self.load_dataset(data_path, with_label = False)
        predict_ds = self.construct_dataset(dataset, False)
        predict_stream = self.construct_sequencial_stream(predict_ds)
        return predict_stream, dataset

    def construct_dataset(self, dataset, with_label = True):
        dataset = zip(*dataset)
        pairs = []
        if with_label:
            for i in range(len(self.provide_souces)):
                pairs.append((self.provide_souces[i], dataset[i]))
            return IndexableDataset(indexables = OrderedDict(pairs))
        else:
            for i in range(len(self.provide_souces)-1):
                pairs.append((self.provide_souces[i], dataset[i]))
            return IndexableDataset(indexables = OrderedDict(pairs))

    def construct_shuffled_stream(self, dataset):
        it = ShuffledExampleScheme(dataset.num_examples)
        stream = DataStream(dataset, iteration_scheme=it)
        # Sort sets of multiple batches to make batches of similar sizes
        stream = Batch(stream, iteration_scheme=ConstantScheme(self.config.batch_size * self.config.sort_batch_count))
        comparison = _balanced_batch_helper(stream.sources.index('context'))
        stream = Mapping(stream, SortMapping(comparison))
        stream = Unpack(stream)
        stream = Batch(stream, iteration_scheme=ConstantScheme(self.config.batch_size))
        # Add mask
        stream = Padding(stream, mask_sources=['context'], mask_dtype= self.config.int_type)
        return stream

    def construct_sequencial_stream(self, dataset):
        it = SequentialScheme(dataset.num_examples, self.config.batch_size)
        stream = DataStream(dataset, iteration_scheme=it)
        # Add mask
        stream = Padding(stream, mask_sources=['context'], mask_dtype= self.config.int_type)
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
                        dataset.append(self.parse_one_sample(line, with_label))
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
        if self.config.develop: # Assuming that during developing, all the data has been pre-processed                
            array = line.split('\t')
            mention_tokens = tokenize(array[0])                                
            context_tokens = tokenize(array[len(array)-1])
        else:
            line = split_hyphen(line.strip())
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
                if (word not in self.word_freq) or (self.word_freq[word] < 50):
                    context_tokens[i] = self.stem(context_tokens[i])                         
            elif (word not in self.word_freq) or (self.word_freq[word] < 10):
                context_tokens[i] = self.stem(context_tokens[i])                         
        for i in range(len(mention_tokens)):
            word = mention_tokens[i].decode('utf-8').lower()
            if (word not in self.word_freq) or (self.word_freq[word] < 50):
                mention_tokens[i] = self.stem(mention_tokens[i])
        # Add begin_of_sentence label
        _context = self.to_word_ids(['<BOS>']+context_tokens)
        _mention_begin = numpy.int32(begin).astype(self.config.int_type)
        _mention_end = numpy.int32(end).astype(self.config.int_type)
        mention = array[0]
        context = array[len(array)-1]
        if with_label:
            return (_context, _mention_begin, _mention_end, _label, mention, context)
        else:
            return (_context, _mention_begin, _mention_end, mention, context)

    def get_word2id(self, data_path):
        '''
        Construct word2id table.
        '''

        if os.path.isdir(data_path):
            files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
        else:
            data_path, file = os.path.split(data_path)
            files = [file]
        self.word2id = dict()
        self.word2id['<UNK>'] = len(self.word2id)
        self.word2id['<BOS>'] = len(self.word2id)
        for file in files:
            if os.path.isfile(os.path.join(data_path, file)):
                with codecs.open(os.path.join(data_path, file),"r", "UTF-8", errors = "ignore") as f:
                    for line in f:
                        try:
                            contexts = line.strip().split('\t')[2].split(' ')
                            for i in range(len(contexts)):
                                word = contexts[i].decode('utf-8').lower()
                                if (word not in self.word_freq) or (self.word_freq[word] < 10):
                                    contexts[i] = self.stem(contexts[i])
                            for word in contexts:
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
        print(os.path.abspath(data_path))
        if os.path.isdir(data_path):
            files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
        else:
            data_path, file = os.path.split(data_path)
            files = [file]
        self.word_freq = dict()
        for file in files:
            if os.path.isfile(os.path.join(data_path, file)):
                with codecs.open(os.path.join(data_path, file),"r", "UTF-8", errors = "ignore") as f:
                    for line in f:
                        try:
                            contexts = line.strip().split('\t')[2].split(' ')
                            for word in contexts:
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
    def __init__(self,config):
        '''
        @param config: Model config module
        '''
        super(MTLD, self).__init__(config)
        self.dbpedia = DBpedia()
        self.provide_souces = ('context', 'mention_begin', 'mention_end', 'type', 'type_weight', 'label')
        self.type2id = None
        if os.path.exists(config.type2id_path):
            self.load_type2id()
        else:
            raise Exception("Type2id cannot be None!")

    def parse_one_sample(self, line, with_label = True):
        # Extract mention matched types
        sample = super(MTLD, self).parse_one_sample(line, with_label)
        mention = sample[-2]
        pairs = self.dbpedia.get_match_entities(mention)
        if pairs is not None:
            types, indegrees = zip(*pairs)
            _type = numpy.array([self.type2id[type] for type in types], dtype="int32")
            weights = [numpy.sqrt(indegree) for indegree in indegrees]  # log indegree
            s = sum(weights) # max normalization
            _type_weight = numpy.array([numpy.float32(weight/s) for weight in weights])
        else:
            _type = numpy.array([self.type2id["UNKNOWN"]],dtype="int32")
            _type_weight = numpy.array([1.0], dtype="float32")
        return sample[0:3]+ (_type, _type_weight) + sample[3:]

    def load_type2id(self):
        self.type2id = load_dic(self.config.type2id_path)
