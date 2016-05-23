#-*- coding:utf-8 -*-
import config
import os
import sys
import re

import codecs
import nltk

from os import listdir

nltk.data.path.append(r"D:\Data\NLTK Data")
test_str = "There should be offset information!"
sld = nltk.word_tokenize(test_str)