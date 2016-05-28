'''
Filter out #N# symbols within the context
'''
import config

import os
import sys
import re

regex1 = re.compile(r"(^|\s)([a-zA-Z]+)/([a-zA-Z]+)(?=$|\s)")
regex2 = re.compile(r"(\w+)-(?=\w+)")

def hyphen_filter(samples):
    samples_copy = []
    for sample in samples:
        mention = sample[0]
        context = sample[1]
        mention = regex1.sub("\g<1>\g<2> / \g<3>", mention)
        mention = regex2.sub("\g<1> - ", mention)
        context = regex1.sub("\g<1>\g<2> / \g<3>", context)
        context = regex2.sub("\g<1> - ", context)
        samples_copy.append((mention,context))
    return samples_copy