import codecs
import re
import nltk

from nltk.tokenize.punkt import PunktSentenceTokenizer

nltk.data.path.append(r"D:\Data\NLTK Data")

tokenizer = PunktSentenceTokenizer()

def construct_sample(char_begins, char_ends, context):
    samples = []
    idxs = tokenizer.span_tokenize(context)
    contexts = []
    for begin, end in idxs:
        contexts += [nltk.word_tokenize(context[begin:end])]
    for char_begin, char_end in zip(char_begins,char_ends):
        try:
            mention = context[char_begin:char_end]
            mention_tokens = nltk.word_tokenize(mention)
            context_tokens = None
            count = 0
            for begin, end in idxs:
                if begin <= char_begin and end >= char_end:
                    context_tokens = contexts[count]
                    break
                count += 1
            if context_tokens is None:
                raise Exception("Cannot find mention: %s in given context: %s" %(mention,context))
            samples.append((" ".join(mention_tokens) ," ".join(context_tokens)))
        except Exception as e:
            try:
                print("Error! %s" %e.message)
            except:
                print("Encounter weird character! Skip this sample")
            continue
    return samples


