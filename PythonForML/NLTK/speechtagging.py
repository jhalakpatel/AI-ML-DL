import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer # unsupervised tokenizer

# get the training and sample text into variables
train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

print(train_text)

# define cutsom tokenizer
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print('words', words)
            print('tagged',tagged)
    except Exception as e:
        print(str(e))

process_content()

"""
    Speech tagging tags the words as part of speech such as 
    noun, adjective, verb etc
    Pos_tagging covers also tenses of the part of the speech
"""
