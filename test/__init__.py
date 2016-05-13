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

from dataset import satori
from paramsaveload import SaveLoadParams

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

sys.setrecursionlimit(2000000)

if __name__ == "__main__":
    model_name = "deep_bidir_lstm"
    config = importlib.import_module('.%s' % model_name, 'config')
    # Build datastream
    data_path = config.data_path
    test_path = os.path.join(data_path, "test")

    word2id = None
    if os.path.exists(config.word2id_path):
        word2id = {}
        with codecs.open(config.word2id_path, "r", encoding = "UTF-8") as f:
            for line in f:
                array = line.split('\t')
                word2id[array[0]] = int(array[1])
    else:
        raise("Cannot find vocabulary file!")

    print("Loading test dataset...")
    ds, test_stream = satori.setup_datastream(test_path, config, word2id)
    print("Done!")
    model_path = os.path.join(config.model_path, model_name+"_gpu.pkl")

    # Build model
    m = config.Model(config, ds)

    # Build the Blocks stuff for training
    model = Model(m.sgd_cost)   # build computation graph


    if config.save_freq is not None and dump_path is not None:
        extensions += [
            SaveLoadParams(path=dump_path,
                           model=model,
                           before_training=True,    # if exist model, the program will load it first
                           )
        ]
    if valid_stream is not None and config.valid_freq != -1:
        extensions += [
            DataStreamMonitoring(
                [v for l in m.monitor_vars_valid for v in l],
                test_stream,
                prefix='test',
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

