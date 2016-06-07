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
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal

from lstm import WLSTM, MWLSTM

class MTLM():
    def __init__(self, config, dataset):
        context = tensor.imatrix('context')                                 # shape: batch_size*sequence_length
        context_mask = tensor.imatrix('context_mask')
        mention_begin = tensor.ivector('mention_begin')
        mention_end = tensor.ivector('mention_end')
        label = tensor.ivector('label')
        bricks = []

        # set time as first dimension
        context = context.dimshuffle(1, 0)
        context_mask = context_mask.dimshuffle(1, 0)

        # Initialize embedding
        embed = Lookup(len(dataset.word2id), config.embed_size, name='word_embed')
        embs = initialize_embed(config, dataset.word2id)
        embed.initialize_with_pretrain(embs)                    # initialize embeding table with pre-traing values
        # Embed contexts
        context_embed = embed.apply(context)

        h0 = None
        c0 = None
        # Create and apply LSTM
        for time in range(config.lstm_time):
            lstm_ins = Linear(input_dim=config.embed_size, output_dim=4 * config.lstm_size, name='lstm_in_%s' % time)
            lstm_ins.weights_init = IsotropicGaussian(std= numpy.sqrt(2)/numpy.sqrt(config.embed_size+config.lstm_size))
            lstm = LSTM(dim=config.lstm_size, activation=Tanh(), name='lstm_%s' % time)
            lstm.weights_init = IsotropicGaussian(std= 1/numpy.sqrt(config.lstm_size))
            bricks += [lstm_ins, lstm]
            lstm_tmp = lstm_ins.apply(context_embed)
            if h0 is None:
                lstm_hidden, lstm_cell = lstm.apply(inputs = lstm_tmp, mask=context_mask.astype(theano.config.floatX))
            else:
                lstm_hidden, lstm_cell = lstm.apply(inputs = lstm_tmp, states = h0, lstm_cell = c0, mask=context_mask.astype(theano.config.floatX))
            h0 = lstm_hidden[-1, :, :]
            c0 = lstm_cell[-1,:,:]
        # Create and apply output MLP
        out_mlp = MLP(dims = [config.lstm_size*2] + [config.n_labels],
                          activations = [Identity()],
                          name='out_mlp')
        out_mlp.weights_init = IsotropicGaussian(std = numpy.sqrt(2)/numpy.sqrt(config.lstm_size+config.n_labels))
        bricks.append(out_mlp)
        mention_hidden = tensor.concatenate([lstm_hidden[mention_end, tensor.arange(context.shape[1]), :],
                                            lstm_hidden[mention_begin, tensor.arange(context.shape[1]), :]],axis=1)
        self.mention_hidden = mention_hidden
        probs = out_mlp.apply(mention_hidden)
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
        self.error_rate = error_rate

        # Initialize bricks
        for brick in bricks:
            brick.biases_init = Constant(0)
            brick.initialize()

class MTLDM():
    def __init__(self, config, dataset):
        context = tensor.imatrix('context')                                 # shape: batch_size*sequence_length
        context_mask = tensor.imatrix('context_mask')
        type = tensor.imatrix('type')
        type_weight = tensor.matrix('type_weight', dtype=theano.config.floatX)
        mention_begin = tensor.ivector('mention_begin')
        mention_end = tensor.ivector('mention_end')
        label = tensor.ivector('label')
        bricks = []


        # set time as first dimension
        context = context.dimshuffle(1, 0)
        context_mask = context_mask.dimshuffle(1, 0)

        # Embed contexts
        embed = Lookup(len(dataset.word2id), config.embed_size, name='word_embed')
        embs = initialize_embed(config, dataset.word2id)
        embed.initialize_with_pretrain(embs)                    # initialize embeding table with pre-traing values

        # Embed types
        type_lookup = LookupTable(len(dataset.type2id), config.type_embed_size, name="type_embed")
        type_lookup.weights_init = IsotropicGaussian(std= 1/numpy.sqrt(config.type_embed_size))
        type_lookup.initialize()
        type_embed = (type_lookup.apply(type)*type_weight[:,:,None]).sum(axis=1)

        # Apply embedding
        context_embed = embed.apply(context)

        h0 = None
        c0 = None
        # Create and apply LSTM
        for time in range(config.lstm_time):
            lstm_ins = Linear(input_dim=config.embed_size, output_dim=4 * config.lstm_size, name='lstm_in_%s' % time)
            lstm_ins.weights_init = IsotropicGaussian(std= numpy.sqrt(2)/numpy.sqrt(config.embed_size+config.lstm_size))
            lstm = LSTM(dim=config.lstm_size, activation=Tanh(), name='lstm_%s' % time)
            lstm.weights_init = IsotropicGaussian(std= 1/numpy.sqrt(config.lstm_size))
            bricks += [lstm_ins, lstm]
            lstm_tmp = lstm_ins.apply(context_embed)
            if h0 is None:
                lstm_hidden, lstm_cell = lstm.apply(inputs = lstm_tmp, mask=context_mask.astype(theano.config.floatX))
            else:
                lstm_hidden, lstm_cell = lstm.apply(inputs = lstm_tmp, states = h0, mask=context_mask.astype(theano.config.floatX))
            h0 = lstm_hidden[-1, :, :]
            c0 = lstm_cell[-1,:,:]

        # Create and apply output MLP
        out_mlp = MLP(dims = [config.lstm_size*2+config.type_embed_size] + [config.n_labels],
                          activations = [Identity()],
                          name='out_mlp')
        out_mlp.weights_init = IsotropicGaussian(std = numpy.sqrt(2)/numpy.sqrt(config.lstm_size+config.n_labels))
        bricks.append(out_mlp)
        mention_hidden = tensor.concatenate([lstm_hidden[mention_end, tensor.arange(context.shape[1]), :],
                                            lstm_hidden[mention_begin, tensor.arange(context.shape[1]), :]],axis=1)
        self.mention_hidden = mention_hidden
        mlp_input = tensor.concatenate([mention_hidden,type_embed],axis=1)
        probs = out_mlp.apply(mlp_input)
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
        self.error_rate = error_rate
        self.pred = pred
        # Initialize bricks
        for brick in bricks:
            brick.biases_init = Constant(0)
            brick.initialize()

class WLSTMM():
    def __init__(self, config, dataset):
        context = tensor.imatrix('context')                                 # shape: batch_size*sequence_length
        mention_begin = tensor.ivector('mention_begin')
        mention_end = tensor.ivector('mention_end')
        context_mask = tensor.imatrix('context_mask')
        distance = tensor.imatrix('distance')
        label = tensor.ivector('label')
        delta = theano.shared((10.0*numpy.sqrt(2.0)).astype(theano.config.floatX), name = 'delta')
        self.delta = delta
        weights = tensor.exp(-distance*distance/(delta*delta))
        self.weights = weights
        # set time as first dimension
        context = context.dimshuffle(1, 0)
        context_mask = context_mask.dimshuffle(1, 0)
        weights = weights.dimshuffle(1,0)

        # Initialize embedding
        embed = Lookup(len(dataset.word2id), config.embed_size, name='word_embed')
        embs = initialize_embed(config, dataset.word2id)
        embed.initialize_with_pretrain(embs)                    # initialize embeding table with pre-traing values
        # Embed contexts
        context_embed = embed.apply(context)

        lstm_ins = Linear(input_dim=config.embed_size, output_dim=4 * config.lstm_size, name='lstm_in')
        lstm_ins.weights_init = IsotropicGaussian(std= numpy.sqrt(2)/numpy.sqrt(config.embed_size+config.lstm_size*4))
        lstm_ins.biases_init = Constant(0)
        lstm_ins.initialize()
        mwlst = MWLSTM(times = 2, shared = False, dim = config.lstm_size)
        mwlst.weights_init = IsotropicGaussian(std= 1/numpy.sqrt(config.lstm_size))
        mwlst.biases_init = Constant(0)
        mwlst.initialize()
        mwlstm_hidden, _ = mwlst.apply(inputs = lstm_ins.apply(context_embed), weights = weights, mask=context_mask.astype(theano.config.floatX))

        # Create and apply output MLP
        out_mlp = MLP(dims = [config.lstm_size*2] + [config.n_labels],
                          activations = [Identity()],
                          name='out_mlp')
        out_mlp.weights_init = IsotropicGaussian(std = numpy.sqrt(2)/numpy.sqrt(config.lstm_size*2+config.n_labels))
        out_mlp.biases_init = Constant(0)
        out_mlp.initialize()
        out_mlp_inputs = tensor.concatenate([mwlstm_hidden[mention_end, tensor.arange(context.shape[1]), :],
                                            mwlstm_hidden[mention_begin, tensor.arange(context.shape[1]), :]],axis=1)
        probs = out_mlp.apply(out_mlp_inputs)
        # Calculate prediction, cost and error rate
        pred = probs.argmax(axis=1)
        cost = Softmax().categorical_cross_entropy(label, probs).mean()
        error_rate = tensor.neq(label, pred).mean()

        # Other stuff
        cost.name = 'cost'
        error_rate.name = 'error_rate'

        self.sgd_cost = cost
        self.monitor_vars = [[cost], [error_rate],[delta]]
        self.monitor_vars_valid = [[cost], [error_rate], [delta]]
        self.pred = pred
        self.error_rate = error_rate


def initialize_embed(config, word2id):
    path = config.embed_path
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
                    embs += [(word2id[array[0]], numpy.asarray(vector, theano.config.floatX))]
    return embs
