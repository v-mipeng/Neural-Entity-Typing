import logging
import numpy
import sys
import os

import theano
from theano import tensor

from blocks.extensions import Printing, SimpleExtension, FinishAfter, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent

from dataset.multi_time_lstm import MTL, MTLD
from paramsaveload import SaveLoadParams

from config.multi_time_lstm import MTLC, MTLDC

from abc import abstractmethod, ABCMeta

from base import *
from __builtin__ import super


try:
    from blocks_extras.extensions.plot import Plot
    plot_avail = True
except ImportError:
    plot_avail = False
    print "No plotting extension available."

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

sys.setrecursionlimit(2000000)

from multi_time_lstm import MTLE
from dataset.tridi_lstm import TDLD
from config.tridi_lstm import TDLC

class TDLE(MTLE):
    def __init__(self):
        super(TDLE, self).__init__()

    def init_ds(self):
        self.config = TDLC()
        self.ds = TDLD(self.config)

