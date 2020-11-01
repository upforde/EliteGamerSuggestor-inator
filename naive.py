# Importing the data module
import data
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Create a multiomail naive bayes item
# Feed the naive bayes item top used words from each classification

# Probablity of Spam
prob_spam = data.num_spam / data.num_total
print(prob_spam)

def feature_extraction(df):
    feature_matrix = np.zeros((data.num_total, len(data.words_ham + data.words_ham)))
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, df['label'], test_size=0.33)
    return X_train.shape

feature_extraction(data.get_data())
