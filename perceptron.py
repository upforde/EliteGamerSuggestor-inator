import data, sys
import numpy as np
from sklearn.model_selection import train_test_split

#------------------ function definitions ------------------

def create_dictionary(x, y):
    dictionary = {}
    if len(x) == len(y):
        for i in x.keys():
            dictionary.update({x[i]: y[i]})
    return dictionary

def learn_weights(weights, training_set, num_iterations, learning_constant):
    print(f"Iterating {num_iterations} times with a learning constant of {learning_constant}")
    for i in range(num_iterations):
        for email in training_set.keys():
            weight_sum = 0.0
            for word in email.split():
                if word not in weights.keys(): weights.update({word: 0.0})
                f_spam = words_spam[word] if word in words_spam.keys() else 0.0
                f_ham = words_ham[word] if word in words_ham.keys() else 0.0
                weight_sum += weights[word] * (f_spam-f_ham)
            perceptron_output = 1 if weight_sum > 0 else 0
            for word in email.split():
                f_spam = words_spam[word] if word in words_spam.keys() else 0.0
                f_ham = words_ham[word] if word in words_ham.keys() else 0.0
                weights[word] += float(learning_constant) * float(training_set[email]-perceptron_output) * float((f_spam-f_ham))
             
def apply(weights, instance):
    weight_sum = 0.0
    for word in instance.split():
        if word not in weights.keys(): weights.update({word: 0.0})
        f_spam = words_spam[word] if word in words_spam.keys() else 0.0
        f_ham = words_ham[word] if word in words_ham.keys() else 0.0
        weight_sum += weights[word] * (f_spam-f_ham)
    if weight_sum > 0: return 1
    return 0

def test_weights(weights, testing_set):
    tp, tn, fp, fn = 0, 0, 0, 0
    for word in testing_set.keys():
        guess = apply(weights, word)
        if guess == testing_set[word]:
            if guess == 0: tp += 1
            else: tn += 1
        else:
            if guess == 0: fp += 1
            else: fn += 1
    return tp, tn, fp, fn

#-------------------- running the code --------------------

data.count_words()
df = data.get_data()
words_ham, words_spam = data.order_words()
words_ham, words_spam = dict(words_ham), dict(words_spam)
x_train, x_test, y_train, y_test = train_test_split(df['email'], df['label'], test_size = 0.333)

training_set = create_dictionary(x_train, y_train)
testing_set = create_dictionary(x_test, y_test)

weights = {}
itr, lc = 20, 0.1
if len(sys.argv) >= 2: itr = int(sys.argv[1])
if len(sys.argv) >= 3: lc = float(sys.argv[2])
learn_weights(weights, training_set, itr, lc)

tp, tn, fp, fn = test_weights(weights, testing_set)

print(f"True positives: {tp}\nTrue negatives: {tn}\nFalse positives: {fp}\nFalse negatives: {fn}")

