#!/usr/bin/env python

import logging
import numpy
import sys
import os
import importlib
import codecs

import theano
from theano import tensor

from blocks.extensions import Printing, SimpleExtension, FinishAfter, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent

try:
    from blocks_extras.extensions.plot import Plot
    plot_avail = True
except ImportError:
    plot_avail = False
    print "No plotting extension available."

import dataset
from dataset import satori_multi
from paramsaveload import SaveLoadParams

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

sys.setrecursionlimit(2000000)

if __name__ == "__main__":
    model_name = "multi_time_lstm"
    config = importlib.import_module('.%s' % model_name, 'config')
    # Build datastream
    data_path = config.data_path
    train_path = os.path.join(data_path, "train")
    valid_path = os.path.join(data_path, "valid")

    # Load word2id and word_freq dictionary
    word2id = None
    if os.path.exists(config.word2id_path):
        word2id = {}
        with codecs.open(config.word2id_path, "r", encoding = "UTF-8") as f:
            for line in f:
                array = line.split('\t')
                word2id[array[0]] = int(array[1])
    word_freq = None
    if os.path.exists(config.word_freq_path):
        word_freq = {}
        with codecs.open(config.word_freq_path, "r", encoding = "UTF-8") as f:
            for line in f:
                array = line.split('\t')
                word_freq[array[0]] = int(array[1])
    
    print("Loading training dataset...")
    ds, train_stream = satori_multi.setup_datastream(valid_path, config, word2id, word_freq)
    print("Done!")

    # Save word2id and word_freq
    if  not os.path.exists(config.word2id_path):
        word2id = ds.word2id
        with codecs.open(config.word2id_path, "w+", encoding = "UTF-8") as f:
            for key, value in word2id.iteritems():
                f.write("%s\t%s\n" % (key, value))
    if  not os.path.exists(config.word_freq_path):
        word_freq = ds.word_freq
        with codecs.open(config.word_freq_path, "w+", encoding = "UTF-8") as f:
            for key, value in word_freq.iteritems():
                f.write("%s\t%s\n" % (key, value))

    print("Loading validation dataset...")
    _, valid_stream = satori_multi.setup_datastream(valid_path, config, word2id = word2id, word_freq = word_freq)
    print("Done!")
    dump_path = os.path.join(config.model_path, model_name+"_on_satori.pkl")

    # Build model
    m = config.Model(config, ds)

    # Build the Blocks stuff for training
    model = Model(m.sgd_cost)   # build computation graph

    algorithm = GradientDescent(cost=m.sgd_cost,
                                step_rule=config.step_rule,
                                parameters=model.parameters)

    #region Debug Theano Tensor Variable

    #cg = ComputationGraph(m.sgd_cost)
    #input_names = cg.inputs
    #f_cost = theano.function(inputs = cg.inputs, outputs = m.sgd_cost)
    #f_reverse_context_embed = theano.function(inputs = [item for item in cg.inputs if item.name == "reverse_context"], outputs = m.reverse_context_embed)
    #f_reverse_tmp = theano.function(inputs = [item for item in cg.inputs if item.name == "reverse_context"], outputs = m.bwd_tmp)
    #f_bwd_mask = theano.function(inputs = [item for item in cg.inputs if item.name == "reverse_context_mask"], outputs = m.bwd_mask)
    #f_backward = theano.function(inputs = [item for item in cg.inputs if item.name == "reverse_context" or item.name == "reverse_context_mask"], outputs = m.bwd_hidden)
    #for data in train_stream.get_epoch_iterator():
    #    label = data[train_stream.sources.index('label')]
    #    reverse_context = data[train_stream.sources.index('reverse_context')]
    #    reverse_context_mask = data[train_stream.sources.index('reverse_context_mask')]
    #    order_context = data[train_stream.sources.index('order_context')]
    #    order_context_mask = data[train_stream.sources.index('order_context_mask')]
    #    print(f_reverse_context_embed(reverse_context).shape)
    #    raw_input('continue?')
    #    print(f_reverse_tmp(reverse_context).shape)
    #    raw_input('continue?')
    #    print(f_bwd_mask(reverse_context_mask).shape)
    #    raw_input('continue?')
    #    print(f_backward(order_context, order_context_mask))
    #    print(f_cost(label, reverse_context, reverse_context_mask, order_context, order_context_mask))
            
    #endregion

    extensions = [
            TrainingDataMonitoring(
                [v for l in m.monitor_vars for v in l],
                prefix='train',
                every_n_batches=config.print_freq)
    ]
    if config.save_freq is not None and dump_path is not None:
        extensions += [
            SaveLoadParams(path=dump_path,
                           model=model,
                           before_training=True,    # if exist model, the program will load it first
                           after_training=True,
                           after_epoch=True,
                           every_n_batches=config.save_freq)
        ]
    if valid_stream is not None and config.valid_freq != -1:
        extensions += [
            DataStreamMonitoring(
                [v for l in m.monitor_vars_valid for v in l],
                valid_stream,
                prefix='valid',
                every_n_batches=config.valid_freq),
        ]
    extensions += [
            Printing(every_n_batches=config.print_freq, after_epoch=True),
            ProgressBar()
    ]


    main_loop = MainLoop(
        model=model,
        data_stream=train_stream,
        algorithm=algorithm,    # learning algorithm: AdaDelta, Momentum or others
        extensions=extensions
    )

    # Run the model !
    main_loop.run()
    main_loop.profile.report()

