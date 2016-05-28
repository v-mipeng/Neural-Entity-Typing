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
    word = word.decode('utf-8').encode('ASCII')
    word = regex1.sub("AA", word)
    word = regex2.sub("aa", word)
    word = regex3.sub("00", word)
    return word.encode('utf-8')

basedir = config.basedir
files = listdir(basedir)
for file in files:
    if os.path.isfile(os.path.join(basedir,file)):
        writer = codecs.open(os.path.join(basedir,"temp",file),"w+", 'UTF-8')
        word2time = dict()
        with codecs.open(os.path.join(basedir,file),"r", "UTF-8") as f:
            for line in f:
                try:
                    line = line.strip()
                    array = line.split('\t')
                    mention = array[0]
                    type = array[1]
                    context = array[2]
                    contexts = context.split(' ')
                    for word in contexts:
                        if word in word2time:
                            word2time[word.decode('utf-8').lower()] += 1
                        else:
                            word2time[word.decode('utf-8').lower()] = 1
                except Exception as e:
                    try:
                        print("find error in step 1 %s" % e.message)
                    except:
                        print("find error in step 1")
        with codecs.open(os.path.join(basedir,file),"r", "UTF-8") as f:
            for line in f:
                try:
                    line = line.strip()
                    array = line.split('\t')
                    mentions = array[0].split(' ')
                    type = array[1]
                    contexts = array[2].split(' ')
                    for i in range(len(mentions)):
                        if word2time[mentions[i].decode('utf-8').lower()] < 50:
                            mentions[i] = stem(mentions[i])
                    for i in range(len(contexts)):
                        if word2time[contexts[i].decode('utf-8').lower()] < 50:
                            contexts[i] = stem(contexts[i])
                    writer.write("%s\t%s\t%s\n" % (" ".join(mentions),type, " ".join(contexts)))
                except Exception as e:
                    try:
                        print("find error in step 2 %s" % e.message)
                    except:
                        print("find error in step 2")
                    writer.write("%s\n" % line)
        writer.close()

