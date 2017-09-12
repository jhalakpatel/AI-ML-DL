'''
largest nltk capa corpora - take words - look at syn, antonym, defination and context
'''
from nltk.corpus import wordnet
syns = wordnet.synsets('program')

# synset
print(syns[0].lemmas())
# just the word
print(syns[0].lemmas()[0].name())
# defination
print(syns[0].definition())
# example
print(syns[0].examples())

# basic operations
syns = []
ants = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        #print('l:',l)
        syns.append(l.name())
        if l.antonyms():
            ants.append(l.antonyms()[0].name())

print(set(syns))
print(set(ants))

# sematic similarity 
w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')
# compare the similarity between the two words
# wu and pulmer - paper on semantic similarity
print(w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('car.n.01')
print(w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('proton.n.01')
print(w1.wup_similarity(w2))

# wordnet or syns-set - can be use to do plagarism detector using nltk
# can convert synonyms and antonyms and get the new description
