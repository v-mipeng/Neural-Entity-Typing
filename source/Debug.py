#import importlib
#import sys
#from dataset.multi_time_lstm import MTL
#import os
#model_name = "test"
#from config.multi_time_lstm import MTLC, MTLDC
#config = MTLC()
#config2 = MTLDC()
#print(config.basedir)
#print(config2.basedir)

#dataset = MTL(config)
#test_path = os.path.join(config.data_path, "test/conll/")
#train_stream,valid_stream = dataset.get_train_stream(test_path, 0.1)
#train_samples = 0
#valid_samples = 0
#for train_data in train_stream.get_epoch_iterator():
#    train_samples += len(train_data[0])
#for valid_data in valid_stream.get_epoch_iterator():
#    valid_samples += len(valid_data[0])
#print(train_samples, valid_samples)


#import sys
#import os

#import logging
#import numpy
#import sys
#import os
#import importlib
#import codecs

#import theano
#from theano import tensor
#import pydot_ng
#import graphviz

#from blocks.extensions import Printing, SimpleExtension, FinishAfter, ProgressBar
#from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
#from blocks.graph import ComputationGraph
#from blocks.main_loop import MainLoop
#from blocks.model import Model
#from blocks.algorithms import GradientDescent

#x = tensor.fmatrix('x')
#y = tensor.fmatrix('y')
#z = tensor.dot(x,y)+x+2*y
#f_z = theano.function([x,y],z, on_unused_input = "ignore")
#theano.printing.debugprint(f_z)

from entrance.multi_time_lstm import MTLE

predictor = MTLE()
predictor.train()
  