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

from multi_time_lstm import initialize_embed

class TDLM:
    def __init__(self, config, dataset):
        '''
        Triple direction lstm model
        '''
        to_begin = tensor.imatrix('to_begin')                       #from mention begin-1 to context begin
        to_begin_mask = tensor.imatrix('to_begin_mask')             #from mention end+1 to context end
        to_end = tensor.imatrix('to_end')
        to_end_mask = tensor.imatrix('to_end_mask')
        mention = tensor.imatrix('mention')
        mention_mask = tensor.imatrix('mention_mask')
        label = tensor.ivector('label')

        # set time as first dimension
        to_begin = to_begin.dimshuffle(1, 0)
        to_begin_mask = to_begin_mask.dimshuffle(1, 0)
        to_end = to_end.dimshuffle(1, 0)
        to_end_mask = to_end_mask.dimshuffle(1, 0)
        mention = mention.dimshuffle(1, 0)
        mention_mask = mention_mask.dimshuffle(1, 0)
        # Initialize embedding
        embed = Lookup(len(dataset.word2id), config.embed_size, name='word_embed')
        embs = initialize_embed(config, dataset.word2id)
        embed.initialize_with_pretrain(embs)                    
        # Embed contexts
        to_bein_embed = embed.apply(to_begin)
        to_end_embed = embed.apply(to_end)
        mention_embed = embed.apply(mention)

        input_weight_init = IsotropicGaussian(std= numpy.sqrt(2)/numpy.sqrt(config.embed_size+config.lstm_size))
        lstm_weight_init = IsotropicGaussian(std= 1/numpy.sqrt(config.lstm_size))
        inputs = [to_bein_embed, to_end_embed, mention_embed]
        masks = [to_begin_mask, to_end_mask, mention_mask]
        hidden_outputs = []
        # Create and apply LSTM
        name_surfix = ('to_begin', 'to_end','mention')
        for i in range(3):
            lstm_ins = Linear(input_dim=config.embed_size, output_dim=4 * config.lstm_size, name='lstm_in_%s' % name_surfix[i])
            lstm_ins.weights_init = input_weight_init
            lstm_ins.biases_init = Constant(0)
            lstm = LSTM(dim=config.lstm_size, activation=Tanh(), name='lstm_%s' % name_surfix[i])
            lstm.weights_init = lstm_weight_init
            lstm_ins.initialize()
            lstm.initialize()
            lstm_hidden,_ = lstm.apply(inputs = lstm_ins.apply(inputs[i]), mask=masks[i].astype(theano.config.floatX))
            hidden_outputs.append(lstm_hidden[-1,:,:])
        # Create and apply output MLP
        out_mlp = MLP(dims = [config.lstm_size*3] + [config.n_labels],
                            activations = [Identity()],
                            name='out_mlp')
        out_mlp.weights_init = IsotropicGaussian(std = numpy.sqrt(2)/numpy.sqrt(config.lstm_size+config.n_labels))
        out_mlp.biases_init = Constant(0)
        out_mlp.initialize()
        out_mlp_input = tensor.concatenate([hidden for hidden in hidden_outputs], axis = 1)
        probs = out_mlp.apply(out_mlp_input)
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
