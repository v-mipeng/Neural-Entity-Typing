'''
Filter out #N# symbols within the context
'''
import config

import os
import sys
import re

import codecs

from os import listdir


regex1 = re.compile(r"(^|\s)([a-zA-Z]+)/([a-zA-Z]+)(?=$|\s)")
regex2 = re.compile(r"(\w+)-(?=\w+)")

basedir = config.basedir
if not os.path.isdir(os.path.join(basedir,"hypen")):
    os.makedirs(os.path.join(basedir,"hypen"))
files = listdir(os.path.join(basedir, "sen"))
for file in files:
    if os.path.isfile(os.path.join(basedir, "sen", file)):
        writer = codecs.open(os.path.join(basedir,"hypen",file),"w+", 'UTF-8')
        word2time = dict()
        with codecs.open(os.path.join(basedir,"sen", file),"r", "UTF-8") as f:
            for line in f:
                line = line.strip()
                line = regex1.sub("\g<1>\g<2> / \g<3>", line)
                line = regex2.sub("\g<1> - ", line)
                writer.write("%s\n" % line)
        writer.close()

