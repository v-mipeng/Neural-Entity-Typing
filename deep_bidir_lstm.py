#!/usr/bin/env python
import sys
import logging
import numpy
import os
import importlib
import codecs

import theano

from blocks.graph import ComputationGraph
from blocks.model import Model


try:
    from blocks_extras.extensions.plot import Plot
    plot_avail = True
except ImportError:
    plot_avail = False
    print "No plotting extension available."


from dataset import satori_test
from paramsaveload import SaveLoadParams

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

sys.setrecursionlimit(2000000)

if __name__ == "__main__":
    model_name = "deep_bidir_lstm"
    config = importlib.import_module('.%s' % model_name, 'config')

    # Build test datastream
    test_path = os.path.join(config.data_path, "test")
    # load vocabulary
    if os.path.exists(config.word2id_path):
        word2id = {}
        with codecs.open(config.word2id_path, "r", encoding = "UTF-8") as f:
            for line in f:
                array = line.split('\t')
                word2id[array[0]] = int(array[1])
    else:
        raise("Cannot find vocabulary file!")
    print("Loading test dataset...")
    ds, test_stream = satori_test.setup_datastream(test_path, config, word2id)
    print("Done!")
    model_path = os.path.join(config.model_path, model_name+"_gpu.pkl")

    # Build model
    m = config.Model(config, ds)
    cg = ComputationGraph(m.sgd_cost)

    # Build the Blocks stuff for training
    model = Model(m.sgd_cost)   
    initializer = SaveLoadParams(model_path,model)
    initializer.do_load()
    
    # Build predictor
    cg = ComputationGraph(m.sgd_cost)
    f_pred = theano.function(cg.inputs[1:5],m.pred)

    # Do prediction and write the result to file
    des = os.path.join("./output/result/", model_name,".txt");
    writer = codecs.open(des,"w+")
    print("Predicting...")
    for inputs in test_stream.get_epoch_iterator():
        labels = f_pred(inputs[test_stream.sources.index("label")],
                        inputs[test_stream.sources.index("reverse_context")],
                        inputs[test_stream.sources.index("reverse_context_mask")],
                        inputs[test_stream.sources.index("order_context")],
                        inputs[test_stream.sources.index("order_context_mask")])
        for lable in labels:
            writer.write(lable+"\n")
    writer.close()
    print("Done!")
