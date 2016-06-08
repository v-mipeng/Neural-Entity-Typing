class BaseClass(object):
    def __init__(self):
        print("base init")
        self.d = 0
        BaseClass.static_fun()

    def test(self):
        self.__sub_fun()

    def _sub_fun(self, value = True):
        print("in base fun")
    
    @classmethod
    def static_fun(cls):
        if cls is BaseClass:
            print("invoke by Base")
        else:
            print("invoke by Sub")
        print("static fun")

class SubClass(BaseClass):
    def __init__(self):
        super(SubClass, self).__init__()
        self.d = 3

    def _sub_fun(self, value = True):
        print("in sub fun")
        return 3,2

def average(values):
    """Computes the arithmetic mean of a list of numbers.

    >>> average([20, 30, 70])
    40
    """
    return 1.0*sum(values) / len(values)

c = SubClass()
g = c._sub_fun()

#import doctest
#doctest.testmod()   # automatically validate the embedded tests
#c = SubClass()
#c.sub_fun()

#region Test Grad
#from blocks.initialization import IsotropicGaussian
#import blocks.initialization
#import theano.tensor as tensor
#from blocks.bricks.recurrent import LSTM
#from blocks.bricks import Tanh
#import numpy
#import theano
#w = theano.shared(numpy.random.rand(3,3),name = "weight")
#w_mask = theano.shared(numpy.ones((3,3)), name = 'weight_mask')
#mask_value = w_mask.get_value()
#mask_value[0,:] = 0
#w_mask.set_value(mask_value)
#w2 = theano.shared(numpy.random.rand(3),name = "weight2")
#x = tensor.vector('x', dtype = theano.config.floatX)
#y = tensor.dot(w2,tensor.dot(w,x))
#grad = theano.grad(y, [w,w2], known_grads = {w[0]:tensor.zeros((3))})
##grad[0] = grad[0]*w_mask
#f_grad = theano.function([x],grad)
#f_y = theano.function([x], y, updates = [(w, w-grad[0]),(w2, w2-grad[1])])
#n_x = numpy.ones(3)
#n_grad = f_grad(n_x)
#n_y = f_y(n_x)
#print(n_grad)
#print('\n')
#print(n_y)
#print('\n')
#n_y = f_y(n_x)
#print('\n')
#print(n_y)
#endregion

#region Test Decorate
def application(fun):
    def new_fun(inputs):
        for input in inputs:
            fun(input)
    return new_fun

@application
def apply(inputs):
    print(inputs)

apply(range(10))
#endregion