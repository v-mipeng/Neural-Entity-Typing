import logging

import numpy

import os

import cPickle


model_path = os.path.join("./","output/models/multi_time_lstm2.pkl");
with open(model_path, 'rb') as f:
    param = cPickle.load(f)
    