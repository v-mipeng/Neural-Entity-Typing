'''
Count word frequency
'''

import config
import os
import sys
import codecs

from os import listdir


basedir = config.basedir
files = listdir(os.path.join(basedir, "hypen"))
word2time = dict()
for file in files:
    if os.path.isfile(os.path.join(basedir, "hypen", file)):
        with codecs.open(os.path.join(basedir, "hypen", file),"r", "UTF-8") as f:
            for line in f:
                line = line.strip()
                contexts = line.split('\t')[2].split(' ')
                for word in contexts:
                    word = word.decode('utf-8').lower()
                    if word in word2time:
                        word2time[word] += 1
                    else:
                        word2time[word] = 1
writer = open(os.path.join(basedir,"word freq.txt"), "w+")
for item in word2time.items():
    writer.write("%s\t%s\n" % (item[0], item[1]))
writer.close()