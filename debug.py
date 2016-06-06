class BaseClass(object):
    def __init__(self):
        print("base init")
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

    def _sub_fun(self, value = True):
        print("in sub fun")

def average(values):
    """Computes the arithmetic mean of a list of numbers.

    >>> average([20, 30, 70])
    40
    """
    return 1.0*sum(values) / len(values)

#import doctest
#doctest.testmod()   # automatically validate the embedded tests
#c = SubClass()
#c.sub_fun()

from blocks.initialization import IsotropicGaussian
import blocks.initialization
import theano.tensor as tensor
from blocks.bricks.recurrent import LSTM
from blocks.bricks import Tanh
import numpy
import theano
w = theano.shared(numpy.random.rand(3,3),name = "weight")
w2 = theano.shared(numpy.random.rand(3),name = "weight2")
x = tensor.vector('x', dtype = theano.config.floatX)
y = tensor.dot(w2,tensor.dot(w,x))
grad = theano.grad(y, [w,w2], consider_constant = [w])
f_grad = theano.function([x],grad)
n_x = numpy.ones(3)
n_grad = f_grad(n_x)
print(w.get_value())
print(n_grad)