import data, sys
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

#------------------ function definitions ------------------
# Function that merges the email and lable tuples into a dictionary
def create_dictionary(x, y):
    dictionary = {}
    if len(x) == len(y):
        for i in x.keys():
            dictionary.update({x[i]: y[i]})
    return dictionary

# Function for perceptron training
def learn_weights(weights, training_set, num_iterations, learning_constant, threshold = 0):
    print(f"The training iterates {num_iterations} times with a learning constant of {learning_constant}\n")
    least_sum = float('inf')
    most_sum = float('-inf')
    # Iterating the weight adjustment process
    for i in range(num_iterations):
        # For each instance in the training set
        for instance in training_set.keys():
            weight_sum = 0.0
            # Loop through each word in the instance
            for word in instance.split():
                # Add the word to the weights dictionary if it's not present
                if word not in weights.keys(): weights.update({word: 0.0})
                # Find the ham and spam frequency for this word in the datasett
                f_spam = words_spam[word] if word in words_spam.keys() else 0.0
                f_ham = words_ham[word] if word in words_ham.keys() else 0.0
                # Add the weighted frequency of ham and/or spam to the sum. If the
                # word is more frequent in ham than spam, then the value will be negative
                weight_sum += weights[word] * (f_spam-f_ham)
            # The perceptron is excited (1) when the sum of weights is more than
            # 0 (spam) and not excited if the sum of weights is less than  or 
            # equal to 0 (not spam)
            perceptron_output = 1 if weight_sum > 0 else 0
            # Update the weights of the relavant words to the instance at hand
            # based on the perceptron training rule
            for word in instance.split():
                f_spam = words_spam[word] if word in words_spam.keys() else 0.0
                f_ham = words_ham[word] if word in words_ham.keys() else 0.0
                weights[word] += float(learning_constant) * float(training_set[instance]-perceptron_output) * float((f_spam-f_ham))
             
# Function to apply the weights to an instance
def apply(weights, instance):
    weight_sum = 0.0
    # Works similarly to the learning function, where the sum of the
    # weights gets summed up, and if the value is positive, then the
    # perceptron is excited (1)
    for word in instance.split():
        if word not in weights.keys(): weights.update({word: 0.0})
        f_spam = words_spam[word] if word in words_spam.keys() else 0.0
        f_ham = words_ham[word] if word in words_ham.keys() else 0.0
        weight_sum += weights[word] * (f_spam-f_ham)
    if weight_sum > 0: return 1
    return 0

# Function to test the weights on a testing set
def test_weights(weights, testing_set):
    tp, tn, fp, fn = 0, 0, 0, 0
    # Each instance in the testing set has the learned weights
    # applied to it.
    for instance in testing_set.keys():
        guess = apply(weights, instance)
        # The perceptron result is then checked with the labeling
        if guess == testing_set[instance]:
            # If the perceptron result is correct, then either the
            # true positive or true negative value is incremented
            if guess == 0: tp += 1
            else: tn += 1
        else:
            # If the perceptron result is incorrect, then either
            # the false positive or false negative value is
            # incremented
            if guess == 0: fp += 1
            else: fn += 1
    return tp, tn, fp, fn

# Function to set the iteration and learning constant parameters
def set_params():
    # Setting the training parameters to be 20 iterations and a learning constant 
    # of 0.1 by default. These may be changed by providing them when starting the 
    # python script
    itr, lc = 20, 0.1
    if len(sys.argv) >= 2: itr = int(sys.argv[1])
    if len(sys.argv) >= 3: lc = float(sys.argv[2])
    return itr, lc

def plot_cm(cm):
    df_cm = pd.DataFrame(cm, index = ["Real Ham", "Real Spam"], columns = ["Guessed Ham", "Guessed Spam"])
    sn.heatmap(df_cm, annot=True, fmt="d", cmap=sn.color_palette("rocket_r", as_cmap=True))
    plt.show()

def plot_roc(num_points, lowest_threshold, highest_threshold):
    print("Creating the ROC graph. This may take a while...")
    weights = {}
    itr, lc = set_params()


#-------------------- running the code --------------------
# Getting the dataset from data.py
data.count_words()
df = data.get_data()
words_ham, words_spam = data.order_words()
x_train, x_test, y_train, y_test = train_test_split(df['email'], df['label'], test_size = 0.333)
# Converting the data from data.py into dictionaries. This choice was done mainly
# because the perceptron seemed easier to implement with the use of dictionaries.
words_ham, words_spam = dict(words_ham), dict(words_spam)
training_set = create_dictionary(x_train, y_train)
testing_set = create_dictionary(x_test, y_test)

# The weight dictionary, as well as the iteration nummber and the learning 
# constant are instanciated, before being sent into the learn_weights function
weights = {}
# Setting the training parameters to be 20 iterations and a learning constant 
# of 0.1 by default. These may be changed by providing them when starting the 
# python script
itr, lc = 20, 0.1
if len(sys.argv) >= 2: itr = int(sys.argv[1])
if len(sys.argv) >= 3: lc = float(sys.argv[2])

# Running the training algorithm with the provided parameters.
learn_weights(weights, training_set, itr, lc)

# The test function returns the four values ov a confusion matrix
tp, tn, fp, fn = test_weights(weights, testing_set)

# Calculating the accuracy, sensitivity and specificity of the method
acc = int((tp+tn)/(tp+tn+fp+fn)*100)
sens = tp/(tp+fn)
spec = fp/(fp+tn)

# Printing the data to terminal
print(f"Accuracy of the model: {acc}%")
print(f"Sensitivity: {sens}")
print(f"Specificity: {spec+1}")

# Plotting the confusion matrix
df_cm = pd.DataFrame([[tp, fp],[fn, tn]], index = ["Real Ham", "Real Spam"], columns = ["Guessed Ham", "Guessed Spam"])
sn.heatmap(df_cm, annot=True, fmt="d", cmap=sn.color_palette("rocket_r", as_cmap=True))
plt.show()