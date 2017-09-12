# stop words - dont care words while data preprocessing - like common words
# such as "a", "the", "an"

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

s1 = "this is an example showing off stop word filteration."

# common words in english - acting as stop words
stop_words = set(stopwords.words('English'))

words = word_tokenize(s1)

'''
s1_filtered = []
for w in words:
    if w not in stop_words:
        s1_filtered.append(w)


'''

s1_filtered = [w for w in words if not w in stop_words]

print(s1_filtered)
