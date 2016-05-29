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


from dataset import satori_multi_test as satori_multi
from paramsaveload import SaveLoadParams

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

sys.setrecursionlimit(2000000)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = numpy.exp(x)
    return e_x / e_x.sum(axis = 0)

if __name__ == "__main__":
    model_name = "multi_time_lstm"
    config = importlib.import_module('.%s' % model_name, 'config')

    # load word2id and word_freq
    if os.path.exists(config.word2id_path):
        word2id = {}
        with codecs.open(config.word2id_path, "r", encoding = "UTF-8") as f:
            for line in f:
                array = line.split('\t')
                word2id[array[0]] = int(array[1])
    else:
        raise("Cannot find vocabulary file!")
        
    if os.path.exists(config.word_freq_path):
        word_freq = {}
        with codecs.open(config.word_freq_path, "r", encoding = "UTF-8") as f:
            for line in f:
                array = line.split('\t')
                word_freq[array[0]] = int(array[1])
    else:
        raise("Cannot find word frequency file!")


    model_path = os.path.join(config.model_path, model_name+".pkl")
    # Build model
    m = None
    f_pred = None
    f_error_rate = None
    pred_inputs = None
    probs_inputs = None
    error_rate_inputs = None
    model = None


    # Do prediction and write the result to file

    label2id = config.to_label_id
    id2label = {
    0:"other",
    1:"location",
    2:"organization",
    3:"person",
    4:"product"
    }
    test_pathes = [os.path.join(config.data_path, "temp/neg_pos.txt"),os.path.join(config.data_path, "temp/pos_neg.txt")]
    result_pathes = [str("./output/result/train on satori bbn and conll/%s test on neg_pos.txt" % model_name), str("./output/result/train on satori bbn and conll/%s test on pos_neg.txt" % model_name)]
    for test_path, result_path in zip(test_pathes,result_pathes):
        writer = codecs.open(result_path,"w+")
        print("Loading test dataset...")
        ds, test_stream = satori_multi.setup_datastream(test_path, config, word2id, word_freq)
        
        if m is None:
            m = config.Model(config, ds)
            model = Model(m.sgd_cost)   
            initializer = SaveLoadParams(model_path,model)
            initializer.do_load()
            cg = ComputationGraph(m.pred)
            f_pred = theano.function(cg.inputs,m.pred)
            pred_inputs = cg.inputs
            cg = ComputationGraph(m.probs)
            probs_inputs = cg.inputs
            f_probs = theano.function(cg.inputs, m.probs)
            cg = ComputationGraph(m.error_rate)
            f_error_rate = theano.function(cg.inputs, m.error_rate)
            error_rate_inputs = cg.inputs
        
        samples = 0
        error_rate = 0
        offset = 0
        print("Predicting...")
        for inputs in test_stream.get_epoch_iterator():
            input_len = len(inputs[test_stream.sources.index(pred_inputs[0].name)])
    
            label_ids = f_pred(inputs[test_stream.sources.index(pred_inputs[0].name)],
                               inputs[test_stream.sources.index(pred_inputs[1].name)],
                               inputs[test_stream.sources.index(pred_inputs[2].name)],
                               inputs[test_stream.sources.index(pred_inputs[3].name)])
            label_probs = f_probs(inputs[test_stream.sources.index(probs_inputs[0].name)],
                               inputs[test_stream.sources.index(probs_inputs[1].name)],
                               inputs[test_stream.sources.index(probs_inputs[2].name)],
                               inputs[test_stream.sources.index(probs_inputs[3].name)])
            error_rate += f_error_rate(inputs[test_stream.sources.index(error_rate_inputs[0].name)],
                                      inputs[test_stream.sources.index(error_rate_inputs[1].name)],
                                      inputs[test_stream.sources.index(error_rate_inputs[2].name)],
                                      inputs[test_stream.sources.index(error_rate_inputs[3].name)],
                                      inputs[test_stream.sources.index(error_rate_inputs[4].name)])*input_len
            samples += input_len
            for true_label_id, label_id, probs, mention, context in zip(inputs[test_stream.sources.index("label")],label_ids, label_probs, ds.mention[offset:offset+input_len], ds.context[offset:offset+input_len]):
                writer.write("%s\t%s\t%s\t%s\t%s\n" % (mention,id2label[int(true_label_id)], id2label[label_id], (numpy.sort(softmax(probs))[-2:]).var(), context))
            offset = offset+input_len
        writer.write("Error rate: %s" % (error_rate/samples))
        writer.close()
        print("Done!")
