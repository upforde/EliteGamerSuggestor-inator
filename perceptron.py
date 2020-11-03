import data, re, string
import numpy as np
from sklearn.model_selection import train_test_split

data.count_words()
df = data.get_data()
words_spam, words_ham = data.words_spam, data.words_ham
x_train, x_test, y_train, y_test = train_test_split(df['email'], df['label'], test_size = 0.333)

"""
def learn_weights(weights, learning_constant, training_set, num_iterations):
    for i in num_iterations:
        for d in training_set:
            weight_sum = weights['weight_zero']
            for f in training_set[d]
"""