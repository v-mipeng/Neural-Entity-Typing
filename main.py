import predictor
import codecs

from dataset import json_script, label_script, ner, english_script, hypen_script

source_path = r"D:\Codes\Project\EntityTyping\Neural Entity Typing\input\new offering\newoffering.txt"
result_path = r"D:\Codes\Project\EntityTyping\Neural Entity Typing\input\new offering\result.txt"
writer = open(result_path,"w+")
with codecs.open(result_path, "w+","UTF-8") as writer:
    with codecs.open(source_path,"r",encoding = "UTF-8") as f:
        for line in f:
            line = line.strip()
            json_doc = ner.ner(line)
            mentions = json_script.parse_json(json_doc)
            char_begins, char_ends, mention_texts= zip(*mentions)
            samples = label_script.construct_sample(char_begins,char_ends,line)
            samples = english_script.eng_filter(samples)
            samples = hypen_script.hyphen_filter(samples)
            mentions, contexts, labels = predictor.predict(samples)
            for mention, label, context in zip(mentions, labels, contexts):
                writer.write("%s\t%s\t%s\n" %(mention, label, context))
