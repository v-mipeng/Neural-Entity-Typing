'''
Filter out #N# symbols within the context
'''
import config

import os
import sys
import re

import codecs

from os import listdir


regex1 = re.compile(r"\|.*\|.*\|")
regex2 = re.compile(r"#N#.*$")
regex3 = re.compile(r"\s*#R#\s*$")      # to .
regex4 = re.compile(r"[^\t]*##\s*") # to \t

basedir = config.basedir
if not os.path.isdir(os.path.join(basedir,"sym")):
    os.makedirs(os.path.join(basedir,"sym"))
files = listdir(os.path.join(basedir, "english"))
for file in files:
    if os.path.isfile(os.path.join(basedir, "english", file)):
        writer = codecs.open(os.path.join(basedir,"sym",file),"w+", 'UTF-8')
        word2time = dict()
        with codecs.open(os.path.join(basedir,"english", file),"r", "UTF-8") as f:
            for line in f:
                line = line.strip()
                if len(regex1.findall(line)) > 0:
                    continue
                line = regex2.sub("",line)
                line = regex3.sub(".",line)
                line = regex4.sub("\t",line)
                writer.write("%s\n" % line)
        writer.close()

