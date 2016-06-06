import numpy
import blocks.bricks
import blocks.bricks.attention
import blocks.bricks.interfaces
from blocks.bricks.recurrent import LSTM, BaseRecurrent, Bidirectional, GatedRecurrent
from blocks.bricks import Tanh, Softmax, Linear, MLP, Identity, Rectifier
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal
from blocks.bricks.base import application

class MLSTM(BaseRecurrent):
    def __init__(self, n_times, input_dim, lstm_dim, init_state, shared = False):
        assert n_times > 1
        self.n_times = n_times
        self.mlstm_ins = Linear(input_dim = input_dim, output_dim=4 * lstm_dim, name='mlstm_ins')
        self.mlstm_ins.weights_init = IsotropicGaussian(std= numpy.sqrt(2)/numpy.sqrt(config.embed_size+config.lstm_size))

        self.lstms = [LSTM(dim = lstm_dim, activation = Tanh(), name = 'lstm_%s' % time) for time in range(self.n_times)]
        if init_state is not None:
            self.lstms[0] = LSTM(dim = lstm_dim, activation = Tanh(), init_state = init_state, name = 'lstm_0')
        for lstm in self.lstms:
            lstm.weights_init = IsotropicGaussian(std= 1/numpy.sqrt(lstm_dim))

    def _allocate(self):
        for lstm in self.lstms:
            lstm.allocate()



    def _initialize(self):
        for lstm in self.lstms:
            lstm.initialize()

    def get_dim(self, name):
        return self.lstms[0].get_dim(name)

    @recurrent(sequences=['inputs', 'mask'], states=['states', 'cells'],
               contexts=[], outputs=['states', 'cells'])
    def apply(self, inputs, states, cells, mask = None):
        app(inputs, mask)
        return self.lstms[-1].apply(inputs, states, cells, mask)
        

    def app(self, inputs, mask):
        for time in range(self.n_time - 1):
            lstm = self.lstms[time]
            lstm_tmp = lstm_ins.apply(inputs)
            if h0 is None:
                lstm_hidden, _ = lstm.apply(inputs = lstm_tmp, mask = mask)
            else:
                lstm_hidden, _ = lstm.apply(inputs = lstm_tmp, states = h0, mask = mask)
            h0 = lstm_hidden[-1, :, :]
        self.lstms[-1] = LSTM(dim = self.lstms[-1].dim, activation = self.lstms[-1].activation, init_state = h0, name = self.lstms[-1].name)
