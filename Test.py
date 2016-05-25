import codecs
import re
import nltk

from nltk.tokenize.punkt import PunktSentenceTokenizer

nltk.data.path.append(r"D:\Data\NLTK Data")

tokenizer = PunktSentenceTokenizer()
regex = re.compile(r"\((\d+),(\d+),[^)]+\)")

writer = codecs.open(r"D:\Codes\Project\EntityTyping\Neural Entity Typing\input\new offering\newoffering to label.txt","w+",)

def construct_dataset(char_begins, char_ends, context):
    idxs = tokenizer.span_tokenize(context)
    contexts = []
    for begin, end in idxs:
        contexts += [nltk.word_tokenize(context[begin:end])]
    for char_begin, char_end in zip(char_begins,char_ends):
        try:
            mention = context[char_begin,char_end]
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
            writer.write("%s\t%s\n" %(mention," ".join(context_tokens)))
        except Exception as e:
            print("Error! %s" %e.message)
            continue


with codecs.open(r"D:\Codes\Project\EntityTyping\Neural Entity Typing\input\new offering\newoffering mention.txt","r") as f:
    for line in f:
        array = line.split('\t')
        context = array[len(array)-1]
        char_begins = []
        char_ends = []
        for i in range(len(array)-1):
            match = regex.match(array[i])
            char_begins.append(int(match.group(1)))
            char_ends.append(int(match.group(2)))
            construct_dataset(char_begins,char_ends,context)

