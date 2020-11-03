# Importing the data module
import data
import pandas as pd
import matplotlib.pyplot as plt


# sklearn library
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import svm

data.count_words(True, True, True)

words_ham, words_spam = data.order_words()

if True:

    print("Top Ham Words: ")
    print("Total number of extracted ham words: " + str(len(words_ham)))
    for i in range(1,100):
        print(str(i) + ".   " + words_ham[i][0] + " - " + str(words_ham[i][1]) )
        
    print("\nTop Spam Words: \n")
    print("Total number of extracted spam words: " + str(len(words_spam)))
    for i in range(1,100):
        print(str(i) + ".   " + words_spam[i][0] + " - " + str(words_spam[i][1]))

