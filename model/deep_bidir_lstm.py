import theano
from theano import tensor
import numpy
import codecs

from blocks.bricks import Tanh, Softmax, Linear, MLP, Identity, Rectifier
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import LSTM

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph, apply_dropout, apply_noise
from word_emb import Lookup


class Model():
    def __init__(self, config, dataset):
        order_context = tensor.imatrix('order_context')                                 # shape: batch_size*sequence_length
        reverse_context = tensor.imatrix('reverse_context')
        order_context_mask = tensor.imatrix('order_context_mask')
        reverse_context_mask = tensor.imatrix('reverse_context_mask')
        label = tensor.ivector('label')
        bricks = []


        # set time as first dimension
        order_context = order_context.dimshuffle(1, 0)
        order_context_mask = order_context_mask.dimshuffle(1, 0)
        reverse_context = reverse_context.dimshuffle(1, 0)
        reverse_context_mask = reverse_context_mask.dimshuffle(1, 0)

        # Embed contexts
        embed = Lookup(dataset.vocab_size, config.embed_size, name='word_embed')
        embs = initialize_embed(config, dataset)
        embed.initialize_with_pretrain(embs)                    # initialize embeding table with pre-traing values
        # Apply embedding
        order_context_embed = embed.apply(order_context)
        reverse_context_embed = embed.apply(reverse_context)

        # Create and apply LSTM
        fwd_lstm_ins = Linear(input_dim=config.embed_size, output_dim=4 * config.lstm_size, name='fwd_lstm_in')
        fwd_lstm = LSTM(dim=config.lstm_size, activation=Tanh(), name='fwd_lstm')

        bwd_lstm_ins = Linear(input_dim=config.embed_size, output_dim=4 * config.lstm_size, name='bwd_lstm_in')
        bwd_lstm = LSTM(dim=config.lstm_size, activation=Tanh(), name='bwd_lstm')

        bricks += [fwd_lstm, bwd_lstm, fwd_lstm_ins, bwd_lstm_ins]

        fwd_tmp = fwd_lstm_ins.apply(order_context_embed)
        bwd_tmp = fwd_lstm_ins.apply(reverse_context_embed)
        fwd_hidden, _ = fwd_lstm.apply(fwd_tmp, mask=order_context_mask.astype(theano.config.floatX))
        bwd_hidden, _ = bwd_lstm.apply(bwd_tmp, mask=reverse_context_mask.astype(theano.config.floatX))  

        # Create and apply output MLP
        out_mlp = MLP(dims = [2*config.lstm_size] + [config.n_labels],
                          activations = [Identity()],
                          name='out_mlp')
        bricks.append(out_mlp)

        probs = out_mlp.apply(tensor.concatenate([fwd_hidden[-1,:,:],bwd_hidden[-1,:,:]], axis=1))

        # Calculate prediction, cost and error rate
        pred = probs.argmax(axis=1)
        cost = Softmax().categorical_cross_entropy(label, probs).mean()
        error_rate = tensor.neq(label, pred).mean()

        # Other stuff
        cost.name = 'cost'
        error_rate.name = 'error_rate'

        self.sgd_cost = cost
        self.monitor_vars = [[cost], [error_rate]]
        self.monitor_vars_valid = [[cost], [error_rate]]
        self.pred = pred

        # Initialize bricks
        for brick in bricks:
            brick.weights_init = config.weights_init
            brick.biases_init = config.biases_init
            brick.initialize()

def initialize_embed(config, dataset):
    path = config.embed_path
    word2id = dataset.word2id
    embs = []
    with codecs.open(path,'r','UTF-8') as f:
        for line in f:
             for line in f:
                word = line.split(' ', 1)[0]
                if word in word2id:
                    array = line.split(' ')
                    if len(array) != config.embed_size + 1:
                        return None
                    vector = []
                    for i in range(1,len(array)):
                        vector.append(float(array[i]))
                    embs += [(word2id[array[0]], numpy.asarray(vector, dtype = theano.config.floatX))]
    return embs

#  vim: set sts=4 ts=4 sw=4 tw=0 et :
