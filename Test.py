words = set()
f= open(r"D:\Codes\Project\EntityTyping\Neural Entity Typing\input\tables\satori and bbn\word2id.txt","r")
for line in f:
    words.add(line.strip().split('\t')[0])
f.close()
writer = open(r"D:\Codes\Project\EntityTyping\Neural Entity Typing\input\tables\word embedding.txt","w+")
count = 1
for line in fileinput.input([r"D:\Data\Google-word2vec\GoogleNews-vectors-negative300.txt"]):
    if count%100 == 0:
        print(count)
    count += 1
    word = line.strip().split(' ',1)[0]
    if word in words:
        writer.write(line)
writer.close()