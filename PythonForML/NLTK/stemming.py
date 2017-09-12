# stemming - retrieving the root word from the actual word - without changing the meaning of the word
# eg. i was ridding in the car == i was taking a ride in the car
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

'''
example_words = ['python', 'pythoner', 'pythoning', 'pythoned', 'pythonly']
for w in example_words:
    print(ps.stem(w))
'''

s1 = 'it is very important to be pythonly while you are pythoning with python. all pythoner have pythoned poorly on python'

word = word_tokenize(s1)

for w in word:
    print(ps.stem(w))
