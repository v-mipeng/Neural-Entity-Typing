'''
Extract word embeddings from pre-trained google word2vec for given words
'''

words = set()
with open(r"D:\Codes\Project\EntityTyping\Neural Entity Typing\input\tables\satori and bbn\word2id.txt","r") as f:
    for line in f:
        words.add(line.strip().split('\t'))
    f.close()
writer = open(r"D:\Codes\Project\EntityTyping\Neural Entity Typing\input\tables\word embedding.txt","w+")
with open(r"D:\Data\Google-word2vec\GoogleNews-vectors-negative300.txt","r") as reader:
    for line in reader:
        word = line.strip().split(' ',1)[0]
        if word in words:
            writer.write(line)
writer.close()