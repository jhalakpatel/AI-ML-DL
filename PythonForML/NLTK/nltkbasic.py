import nltk
#nltk.download()
from nltk.tokenize import sent_tokenize, word_tokenize
example_text = "hello Mr. xyz, how are you doing today? the weather is greate today and python is awesome. the sky is blue and you should not be coding"

# how to split the sentence
#print(sent_tokenize(example_text))
#print(word_tokenize(example_text))

for i in word_tokenize(example_text):
    print(i)

# Motivation:
# computer - can understand text or speech
# how to understand - an article or a story
# organize data - by line or by paragraph 
# basic terms for NLP :
# tokenizer
# word tokenizer 
# sentence tokenizer
# lexicon - dictionary - words and their meanings 
# investor speak -- regular english speak
# 'bull' - positive about the market
# 'bull' - scary animal - in english
# words and meaning depends on the context as well
# corporas - body of text eg. medical journal, presidential speeches, anything in english languages
