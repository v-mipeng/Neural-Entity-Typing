import config
import os
import sys
import re

import codecs

from os import listdir


regex1 = re.compile("[A-Z]+")
regex2 = re.compile("[a-z]+")
regex3 = re.compile("\d+")


def stem(word):
    try:
        word = word.decode('utf-8').encode('ASCII')
    except:
        if regex1.match(word) != None:
            return u"Xx"
        else:
            return u"xx"
    word = regex1.sub("AA", word)
    word = regex2.sub("aa", word)
    word = regex3.sub("00", word)
    return word.encode('utf-8')

basedir = config.basedir

if not os.path.isdir(os.path.join(basedir,"len")):
    os.makedirs(os.path.join(basedir,"len"))
files = listdir(os.path.join(basedir,"hypen"))
for file in files:
    if os.path.isfile(os.path.join(basedir,"hypen",file)):
        writer = codecs.open(os.path.join(basedir,"len", file),"w+", 'UTF-8')
        word2time = dict()
        with codecs.open(os.path.join(basedir,"hypen", file),"r", "UTF-8") as f:
            for line in f:
                line = line.strip()
                array = line.split('\t')
                if len(array[2]) > 5*len(array[0]):
                    writer.write("%s\n" % line)
        writer.close()

