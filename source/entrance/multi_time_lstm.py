import logging
import numpy
import sys
import os

import theano
from theano import tensor

from blocks.extensions import Printing, SimpleExtension, FinishAfter, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent

from dataset.multi_time_lstm import MTL, MTLD
from paramsaveload import SaveLoadParams

from config.multi_time_lstm import MTLC, MTLDC

from abc import abstractmethod, ABCMeta

from base import *


try:
    from blocks_extras.extensions.plot import Plot
    plot_avail = True
except ImportError:
    plot_avail = False
    print "No plotting extension available."

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

sys.setrecursionlimit(2000000)

class BasicEntrance(object):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def train(self, train_path = None, valid_portion = None, model_path = None):
        raise NotImplementedError('subclasses must override train()!')

    @abstractmethod
    def test(self, test_path = None, test_result_path = None, model_path = None):
        raise NotImplementedError('subclasses must override test()!')

    @abstractmethod
    def train(self, predict_path = None, predict_result_path = None, model_path = None):
        raise NotImplementedError('subclasses must override predict()!')

class MTLE(BasicEntrance):
    '''
    Entrance of multiple time LSTM system
    '''
    def __init__(self):
        self.config = None
        self.ds = None
        self.model = None
        self.model_path = None
        self.f_pred = None
        self.m = None
        self.id2label = {
                    0:"other",
                    1:"location",
                    2:"organization",
                    3:"person",
                    4:"product"
                }
        self.init_ds()
    
    def init_ds(self):
        self.config = MTLC()
        self.ds = MTL(self.config)

    def train(self, train_path = None, valid_portion = None, valid_path =None, model_path = None):
        '''
        Train a multi_time_lstm model with given training dataset or the default dataset which is defined with config.multi_time_lstm.BasicConfig.train_path

        @param train_path: path of the training dataset, file or directory, default: config.multi_time_lstm.BasicConfig.train_path
                           File foramt: Mention TAB True_label TAB Context

        @param valid_portion: a float value define the portion of validation, default: config.multi_time_lstm.MLTC.valid_portion
                              size of validation dataset: all_the_sample_num * valid_portion

        @param valid_path: path of the validation dataset, file or directory, if given, the valid_portion will be 0.
                           

        @param model_path: path to dump the trained model, default: config.multi_time_lstm module.model_path
        '''
        if train_path is None:
            train_path = self.config.train_path
        if valid_portion is None:
            valid_portion = self.config.valid_portion
        if model_path is None:
            model_path = self.config.model_path
            self.model_path = model_path
        assert valid_portion >= 0 and valid_portion < 1.0

        if valid_path is None:
            train_stream, valid_stream = self.ds.get_train_stream(train_path, valid_portion)
        else:
            train_stream = self.ds.get_train_stream(train_path, 0.0)
            valid_stream = self.ds.get_train_stream(valid_path, 0.0)
            
        # Build the Blocks stuff for training
        if self.m is None:
            self.m = self.config.Model(self.config, self.ds) # with word2id
        if self.model is None:
            self.model = Model(self.m.sgd_cost) 

        algorithm = GradientDescent(cost=self.m.sgd_cost,
                                    step_rule=self.config.step_rule,
                                    parameters=self.model.parameters,
                                    on_unused_sources='ignore')

        extensions = [
        TrainingDataMonitoring(
            [v for l in self.m.monitor_vars for v in l],
            prefix='train',
            every_n_batches= self.config.print_freq)
            ]

        if self.config.save_freq is not None and model_path is not None:
            extensions += [
                SaveLoadParams(path=model_path,
                                model=self.model,
                                before_training=True,    # if exist model, the program will load it first
                                after_training=True,
                                after_epoch=True,
                                every_n_batches=self.config.save_freq)
            ]
        if valid_stream is not None and self.config.valid_freq != -1:
            extensions += [
                DataStreamMonitoring(
                    [v for l in self.m.monitor_vars_valid for v in l],
                    valid_stream,
    #                before_first_epoch = False,
                    prefix='valid',
                    every_n_batches=self.config.valid_freq),
            ]
        extensions += [
                Printing(every_n_batches=self.config.print_freq, after_epoch=True),
                ProgressBar()
        ]

        main_loop = MainLoop(
            model=self.model,
            data_stream=train_stream,
            algorithm=algorithm,    # learning algorithm: AdaDelta, Momentum or others
            extensions=extensions
        )
        # Run the model !
        main_loop.run()

    def test(self, test_path = None, test_result_path = None, model_path = None):
        '''
        Test with trained multi_time_lstm model on given test dataset

        @param test_path: path of test dataset, file or directory, default: config.multi_time_lstm.BasicConfig.test_path
                          File foramt: Mention TAB True_label TAB Context

        @param test_result_path: path of file or directory to store the test resultk,if not given, the config.multi_time_lstm module.test_result_path will be used
                                 File format: Mention TAB True_label TAB Predict_label TAB Context

        @param model_path: path to load the trained model, default: config.multi_time_lstm.BasicConfig.model_path
        '''
        # Initilize model
        if model_path is not None:
            if self.model_path is None or model_path != self.model_path:
                self.init_model(model_path)
        elif self.model_path is None:
            self.init_model(self.config.model_path)

        # Test
        if test_path is None:
            test_path = self.config.test_path
        if test_result_path is None:
            test_result_path = self.config.test_result_path
        test_files, test_result_files = get_in_out_files(test_path, test_result_path)
        for test_file, test_result_file in zip(test_files,test_result_files):
            print("Test on %s..." % test_file)
            results = self.pred(test_file, for_test = True)
            save_result(test_result_file, results)
            print("Done!")

    def predict(self, predict_path = None, predict_result_path = None, model_path = None):
        '''
        Predicte with trained multi_time_lstm model on given predict dataset
        Output file format: Mention TAB Predict_label TAB Context

        @param predict_path: path of predict dataset, file or directory, default: config.multi_time_lstm.BasicConfig.predict_path
                             File foramt: Mention TAB Context

        @param predict_result_path: path of file or directory to store the predict result, default: config.multi_time_lstm.BasicConfig.predict_result_path
                                    File foramt: Mention TAB Predict_label TAB Context

        @param model_path: path to load the trained model, default: config.multi_time_lstm.BasicConfig.model_path 
        '''
        # Initilize model
        if model_path is not None:
            if self.model_path is None or model_path != self.model_path:
                self.init_model(model_path)
        elif self.model_path is None:
            self.init_model(self.config.model_path)
        # Predict
        if predict_path is None:
            predict_path = self.config.predict_path
        if predict_result_path is None:
            predict_result_path = self.config.predict_result_path
        predict_files, predict_result_files = get_in_out_files(predict_path, predict_result_path)
        for predict_file, predict_result_file in zip(predict_files,predict_result_files):
            print("Predict on %s..." % predict_file)
            results = self.pred(predict_file, for_test = False)
            save_result(predict_result_file, results)
            print("Done!")

    def pred(self, file_path, for_test = True):
        '''
        Make prediction on the samples within given input_file

        @param output_file: result file path, if it is given, write out the predict result into this file

        @param for_test: boolean value, if true, the output format: mention true_label predict_label context
                            otherwise: mention predict_label context

        @return result: a list of tuples, with every tuple consistent with the output format.
        '''
        if for_test:
            stream, data = self.ds.get_test_stream(file_path)
        else:
            stream, data = self.ds.get_predict_stream(file_path)
        cg = ComputationGraph(self.m.pred)
        pred_inputs = cg.inputs
        if self.f_pred is not None:
            f_pred = self.f_pred
        else:
            self.f_pred = f_pred = theano.function(pred_inputs, self.m.pred)  
        result = []
        labels = []
        for inputs in stream.get_epoch_iterator():
            p_inputs = tuple([inputs[stream.sources.index(str(input_name))] for input_name in pred_inputs]) 
            label_ids = f_pred(*p_inputs)
            labels += [self.id2label[label_id] for label_id in label_ids]
        data = zip(*data)
        if for_test:
            true_labels = [self.id2label[label_id] for label_id in data[-3]]
            result = zip(data[-2],true_labels, labels, data[-1])
        else:
            result = zip(data[-2], labels, data[-1])
        return result                 

    def init_model(self, model_path):
        if self.m is None:
            self.m = self.config.Model(self.config, self.ds) # with word2id
        model = Model(self.m.sgd_cost)   
        initializer = SaveLoadParams(model_path, model)
        initializer.do_load()
        self.model = model
        self.model_path = model_path

class MTLDE(MTLE):
    '''
    Entrance of Multiple_Time_LSTM_with_DBpedia system
    '''
    def __init__(self):
        super(MTLDE, self).__init__()

    def init_ds(self):
        self.config = MTLDC()
        self.ds = MTLD(self.config)
