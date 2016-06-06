import numpy
import blocks.bricks
import blocks.bricks.attention
import blocks.bricks.interfaces
from blocks.bricks.recurrent import LSTM, BaseRecurrent, Bidirectional, GatedRecurrent, SimpleRecurrent
from blocks.bricks import Tanh, Softmax, Linear, MLP, Identity, Rectifier
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal
from blocks.bricks.base import application

class WLSTM(LSTM):
    '''
    Weighted LSTM

    Add weights on sequence inputs
    '''
    def __init__(self, *arg, **kwarg):
        super(WLSTM, self).__init__(*arg, **kwarg)

    @recurrent(sequences=['inputs', 'mask', "weights"], states=['states', 'cells'],
               contexts=[], outputs=['states', 'cells'])
    def apply(self, inputs, states = None, cells = None, mask = None, weights = None):
        """Apply Long Short Term Memory transition with elements weighted

        Parameters
        ----------
        states : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current states in the shape
            (batch_size, features). Required for `one_step` usage.
        cells : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current cells in the shape
            (batch_size, features). Required for `one_step` usage.
        inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs in the shape (batch_size,
            features * 4). The `inputs` needs to be four times the
            dimension of the LSTM brick to insure each four gates receive
            different transformations of the input. See [Grav13]_
            equations 7 to 10 for more details. The `inputs` are then split
            in this order: Input gates, forget gates, cells and output
            gates.
        weights: 
            A 1D float array in the shape (batch,) denoting the weights of the elements
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if there is
            data available, 0 if not. Assumed to be 1-s only if not given.

        .. [Grav13] Graves, Alex, *Generating sequences with recurrent
            neural networks*, arXiv preprint arXiv:1308.0850 (2013).

        Returns
        -------
        states : :class:`~tensor.TensorVariable`
            Next states of the network.
        cells : :class:`~tensor.TensorVariable`
            Next cell activations of the network.

        """
        def slice_last(x, no):
            return x[:, no*self.dim: (no+1)*self.dim]

        activation = tensor.dot(states, self.W_state) + inputs
        in_gate = self.gate_activation.apply(
            slice_last(activation, 0) + cells * self.W_cell_to_in)
        forget_gate = self.gate_activation.apply(
            slice_last(activation, 1) + cells * self.W_cell_to_forget)
        next_cells = (
            forget_gate * cells +
            in_gate * self.activation.apply(slice_last(activation, 2)))
        out_gate = self.gate_activation.apply(
            slice_last(activation, 3) + next_cells * self.W_cell_to_out)
        next_states = out_gate * self.activation.apply(next_cells)

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
            next_cells = (mask[:, None] * next_cells +
                          (1 - mask[:, None]) * cells)
        if weights:
            next_cells = next_cells*weights[:, None]*mask[:,None]

        return next_states, next_cells

class MLSTM(LSTM):
    '''
    Multiple time LSTM

    Parameters
    ----------
    dim : int
        The dimension of the hidden state.
    activation : :class:`.Brick`, optional
        The activation function. The default and by far the most popular
        is :class:`.Tanh`.
    gate_activation : :class:`.Brick` or None
        The brick to apply as activation for gates (input/output/forget).
        If ``None`` a :class:`.Logistic` brick is used.    
    times : int
        Times to apply LSTM on the sequence
    shared : bool
        If true, it will apply the same lstm on the input sequence for given times, otherwise
        it will apply "times" different lstm on the input sequence
        
    '''
    def __init__(self, times, shared = False, *args, **kwargs):

        self.times = times
        self.shared = shared
        self.model = LSTM
        super(MLSTM, self).__init__(*args, **kwargs)
        self.init()

    def apply(self, inputs, states = None, cells = None, mask = None):
        h0 = states
        c0 = cells
        for time in range(times):
            if h0 is None:
                lstm_hiddens, lstm_cells = self.rnns[time].apply(inputs = inputs, mask = mask)
            else:
                lstm_hiddens, lstm_cells = self.rnns[time].apply(inputs = inputs, states = h0, cells = c0, mask = mask)
            h0 = lstm_hiddens[-1,:,:]
            c0 = lstm_cells[-1,:,:]   
        return lstm_hiddens, lstm_cells

    def init(self):
        '''
        Initialize multiple lstm
        '''
        self.rnns = []
        if not self.shared:
            for time in range(self.times):
                self.rnns.append(self.model(self.dim, self.activation, self.gate_activation, name = "lstm_%s" %time))
        else:
            self.rnns = [self.model(self.dim, self.activation, self.gate_activation, name = "lstm"]*self.times
            
    def _allocate(self):
        if not self.shared:
            for rnn in self.rnns:
                rnn._allocate()
        else:
            self.rnns[0]._allocate()

    def _initialize(self):
        if not self.shared:
            for rnn in self.rnns:
                rnn._initialize()
        else:
            self.rnns[0]._initialize()

class MWLSTM(MLSTM):
    def __init__(self, times, shared = False, *args, **kwargs):
        return super(MWLSTM, self).__init__(times, shared, *args, **kwargs)
        self.model = MLSTM

    def apply(self, inputs, states = None, cells = None, mask = None, weights = None):
        h0 = states
        c0 = cells
        for time in range(tiems):
            if h0 is None:
                lstm_hiddens, lstm_cells = self.rnns[time].apply(inputs = inputs, weights = weights, mask = mask)
            else:
                lstm_hiddens, lstm_cells = self.rnns[time].apply(inputs = inputs, states = h0, cells = c0, weights = weights, mask = mask)
            h0 = lstm_hiddens[-1,:,:]
            c0 = lstm_cells[-1,:,:]   
        return lstm_hiddens, lstm_cells