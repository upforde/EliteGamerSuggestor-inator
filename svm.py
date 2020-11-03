# Importing the data module
import data
import pandas as pd
import matplotlib.pyplot as plt


# sklearn library
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import svm

data.count_words()

words_ham, words_spam = data.order_words()

if False:

    print("Top Ham Words: ")
    print("Total number of extracted ham words: " + str(len(words_ham)))
    for i in range(1,100):
        print(str(i) + ".   " + words_ham[i][0] + " - " + str(words_ham[i][1]) )
        
    print("\nTop Spam Words: \n")
    print("Total number of extracted spam words: " + str(len(words_spam)))
    for i in range(1,100):
        print(str(i) + ".   " + words_spam[i][0] + " - " + str(words_spam[i][1]))




X_train = data.df.loc[500:2900, 'email'].values
Y_train = data.df.loc[500:2900, 'label'].values


temp = pd.concat( [data.df.loc[:500], data.df.loc[2900:]] )

X_test = temp['email'].values
Y_test = temp['label'].values

vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)
print(train_vectors.shape, test_vectors.shape)

clf = svm.SVC()
clf.fit(X_train, Y_train)

