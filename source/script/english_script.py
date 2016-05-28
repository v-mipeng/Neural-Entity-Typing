'''
mask words Occur less than 
'''
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
if not os.path.isdir(os.path.join(basedir,"english")):
    os.makedirs(os.path.join(basedir,"english"))
files = listdir(os.path.join(basedir, "format"))
for file in files:
    if os.path.isfile(os.path.join(basedir, "format", file)):
        writer = codecs.open(os.path.join(basedir,"english",file),"w+", 'UTF-8')
        with codecs.open(os.path.join(basedir, "format", file),"r", "UTF-8") as f:
            for line in f:
                line = line.strip()
                try:
                    line.decode('utf-8').encode('ascii')
                    writer.write("%s\n" % line)
                except:
                    pass
        writer.close()

