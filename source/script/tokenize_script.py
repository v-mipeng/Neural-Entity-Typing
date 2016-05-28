#-*- coding:utf-8 -*-
import config
import os
import sys
import re

import codecs
import nltk

from os import listdir

nltk.data.path.append(r"D:\Data\NLTK Data")
basedir = config.basedir
if not os.path.isdir(os.path.join(basedir,"sen")):
    os.makedirs(os.path.join(basedir,"sen"))
files = listdir(os.path.join(basedir, "sym"))
for file in files:
    if os.path.isfile(os.path.join(basedir, "sym", file)):
        writer = codecs.open(os.path.join(basedir,"sen",file),"w+", 'UTF-8')
        with codecs.open(os.path.join(basedir, "sym", file),"r", "UTF-8") as f:
            for line in f:
                try:
                    line = line.strip()
                    array = line.split('\t')
                    mention = array[0]
                    type = array[1]
                    context = array[2]
                    mentions = nltk.word_tokenize(mention)
                    contexts = nltk.word_tokenize(context)
                    sentences = nltk.sent_tokenize(context)
                    for sentence in sentences:
                        if sentence.find(mention) != -1:
                            tokens = nltk.word_tokenize(sentence)
                            writer.write("%s\t%s\t%s\n" % (" ".join(mentions),type, " ".join(tokens)))
                            break
                except:
                    print("find error for tokenization!")
        writer.close()