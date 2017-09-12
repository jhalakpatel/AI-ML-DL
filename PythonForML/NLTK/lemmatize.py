from nltk.stem import WordNetLemmatizer
'''
    lemmatizing - better than stemming as it gives actual word with meaning.
    can also club lot of words together.
'''
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('cats'))
print(lemmatizer.lemmatize('better'))
print(lemmatizer.lemmatize('best', pos='a'))
