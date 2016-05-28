'''
Adjust the format of dataset.
mention TAB type TAB context
'''
import config

import os
import sys
import re

import codecs

from os import listdir

basedir = config.basedir
if not os.path.isdir(os.path.join(basedir,"format")):
    os.makedirs(os.path.join(basedir,"format"))
files = listdir(os.path.join(basedir,"raw"))
for file in files:
    if os.path.isfile(os.path.join(basedir,"raw",file)):
        writer = codecs.open(os.path.join(basedir,"format",file),"w+", 'UTF-8')
        with codecs.open(os.path.join(basedir,"raw",file),"r", "UTF-8") as f:
            for line in f:
                try:
                    line = line.strip()
                    array = line.split('\t')
                    if len(array) != 3:
                        writer.write("%s\t%s\t%s\n" % (array[0], array[2], array[3]))
                    else:
                        writer.write("%s\n" % line)
                except:
                    print("Format error in: %s" % line)
        writer.close()

