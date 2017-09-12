import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2005-GWBush.txt')

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            # chunk one or more of anything
            chunkGram = r'''Chunk: {<.*>+}
                                }<VB.?|IN|DT>+{'''
            
            # create a parser based on the chunkgram
            chunkParser = nltk.RegexpParser(chunkGram)

            # pass the tagged images to the parser to the chunks
            chunked = chunkParser.parse(tagged)
            print(chunked)
            chunked.draw()
    except:
        pass


process_content()






'''
    # chinking - chunk everything except from something
'''
