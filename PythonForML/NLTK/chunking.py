import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2005-GWBush.txt')

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            # look for any adverb - zero or more adverbs,verbs zero or more
            # proper noun - 1, and many a noun - this will create a chunk gram
            chunkGram = r'''Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}'''
            
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
    we want to know the meaning of the sentence
    1. who is talking - subject or noun generally
    2. words that modify or affect that noun
        eg. apple releases new phone, comes with covered case, 100$ price. tesla release new battery. - two sentences

    chunks - noun phrases -- group with noun modifiers. we will be using regular expressions.
    you can chunk important words and work from there
'''
