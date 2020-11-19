import data, sys, time
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

#region ------------ function definitions ------------------
# Function that merges the email and lable tuples into a dictionary
def create_dictionary(x, y):
    dictionary = {}
    if len(x) == len(y):
        for i in x.keys():
            dictionary.update({x[i]: y[i]})
    return dictionary

# Function for perceptron training
def learn_weights(weights, training_set, num_iterations, learning_constant, threshold = 0, verbose = True):
    if verbose: print(f"The training iterates {num_iterations} times with a learning constant of {learning_constant}")
    least_sum = float('inf')
    most_sum = float('-inf')
    # Measuring the time it takes to train the model
    t0 = time.time()
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
            # On the last iteration, the lowest and highest weight sum is found.
            # These weight sums will be used to determin the thresholds used for
            # the ROC curve.
            if i == num_iterations-1:
                if weight_sum < least_sum: least_sum = weight_sum
                if weight_sum > most_sum: most_sum = weight_sum
            # The perceptron is excited (1) when the sum of weights is more than
            # 0 (spam) and not excited if the sum of weights is less than  or 
            # equal to 0 (not spam)
            perceptron_output = 1 if weight_sum > threshold else 0
            # Update the weights of the relavant words to the instance at hand
            # based on the perceptron training rule
            for word in instance.split():
                f_spam = words_spam[word] if word in words_spam.keys() else 0.0
                f_ham = words_ham[word] if word in words_ham.keys() else 0.0
                weights[word] += float(learning_constant) * float(training_set[instance]-perceptron_output) * float((f_spam-f_ham))
    # Outputing the time it took to train the model
    if verbose: print("Time spent training the model: %.2f" % (time.time()-t0) + "s.")
    # Function returns the extreme thresholds of the model
    return least_sum, most_sum
             
# Function to apply the weights to an instance
def apply(weights, instance, threshold = 0):
    weight_sum = 0.0
    # Works similarly to the learning function, where the sum of the
    # weights gets summed up, and if the value is positive, then the
    # perceptron is excited (1)
    for word in instance.split():
        if word not in weights.keys(): weights.update({word: 0.0})
        f_spam = words_spam[word] if word in words_spam.keys() else 0.0
        f_ham = words_ham[word] if word in words_ham.keys() else 0.0
        weight_sum += weights[word] * (f_spam-f_ham)
    if weight_sum > threshold: return 1
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
    # Setting parameters to be by default the small dataset of emails,
    # 20 iterations of training, a learning constant of 0.1, the number of points 
    # to be at 10 and number of folds to be 4. These may be changed by
    #  providing them when starting the python script
    big = sys.argv[1].lower() == "big" if len(sys.argv) >= 2 else False
    itr = int(sys.argv[2]) if len(sys.argv) >= 3 else 20
    lc = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.1
    points_num = int(sys.argv[4]) if len(sys.argv) >= 5 else 10
    k_folds = int(sys.argv[5]) if len(sys.argv) >= 6 else 4
    return big, itr, lc, points_num, k_folds

# Function that plots a confusion matrix
def plot_cm(cm):
    df_cm = pd.DataFrame(cm, index = ["Real Ham", "Real Spam"], columns = ["Guessed Ham", "Guessed Spam"])
    sn.heatmap(df_cm, annot=True, fmt="d", cmap=sn.color_palette("rocket_r", as_cmap=True))
    plt.show()

# Function that plots an ROC curve
def plot_roc(ire, lc, num_points, lowest_threshold, highest_threshold, training_set, testing_set):
    print(f"Creating the ROC graph with {num_points} points. This may take a while...")
    # Initiate the arrays that will hold the true positive and true 
    # negative rates
    tp_rates = [0.0]
    fp_rates = [0.0]
    # Calculate the threshold step based on the number of points wanted in
    # the ROC curve
    step = (abs(lowest_threshold) + abs(highest_threshold)) / (num_points+1)
    # Initiate the thresholde to be the lowest threshold
    current_threshold = lowest_threshold
    # Measuring the time it takes to generate an ROC curve
    t0 = time.time()
    # Iterate num_points times to get confusion matricies with different 
    # thresholds
    for _ in range(num_points):
        # Step
        current_threshold += step
        # Create new weights
        weights = {}
        # Run the training function with the new threshold
        learn_weights(weights, training_set, itr, lc, current_threshold, False)
        # Test the new weights and get the confusion matrix for that threshold
        tp, tn, fp, fn = test_weights(weights, testing_set)
        # Append the true positive and false positive rates to the arrays
        tp_rates.append(tp/(tp+fn))
        fp_rates.append(fp/(fp+tn))
    # Finish the curve off at the point 1,1
    tp_rates.append(1)
    fp_rates.append(1)
    # Run the orer_points function to get a nice curve. Without this funciton
    # the curve intersects itself as the thresholds at both low and high extremes
    # produce similar results
    order_points(fp_rates, tp_rates)
    #outputing time it took to generate the ROC curve data
    print("Time spent on generating the ROC curve: %.2f" % (time.time()-t0) + "s.")
    # Plot the curve using matplotlib and show it.
    plt.title('ROC curve')
    plt.plot([0, 1], [0, 1], 'r--', zorder=1)
    plt.plot(fp_rates, tp_rates, color='blue', zorder=2)
    plt.scatter(fp_rates, tp_rates, color='red', zorder=3)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    plt.show()
    
# Dumb function that fixes the order of the points
def order_points(x, y):
    points = []
    # Create pairs of x,y coordinates
    for i in range(0, len(x)):
        points.append([x[i], y[i]])
    # Sort them in ascending x value order
    points.sort()
    # Redistribute the coordinates back to their
    # respective arrays
    for i in range(0, len(x)):
        x[i] = points[i][0]
        y[i] = points[i][1]

# Funciton that preforms cross validation on the perceptron
def cross_validation(k_folds, itr, lc, x_train, y_traing):
    print(f"Performing {k_folds}-fold cross validation.")
    # Measuring the time cross validation takes
    t0 = time.time()
    # Setting up the array that will hold the accuracy scores for each fold
    fold_accuracy = []
    # Splitting the training data into folds
    data = create_folds(k_folds, x_train, y_train)
    # For each fold
    for i in range(k_folds):
        # Setting up new weights for the perceptron, sinece it 
        # has to be refitted with the new data each time
        weights = {}
        # Setting up the dictionaries that will contain the trainig data (which
        # is the rest of the data that is not the fold used for validation)
        # and the validation data (which is the fold used for validation)
        training_set = {}
        validation_set = {}
        for j in range(k_folds):
            if j != i: training_set.update(data[j])
            else: validation_set.update(data[j])
        # Running the training function on the training set
        learn_weights(weights, training_set, itr, lc, 0, False)
        # Testing the weights with the validation fold
        tp, tn, fp, fn = test_weights(weights, validation_set)
        # Appending the accuracy score for this fold
        fold_accuracy.append((tp+tn)/(tp+tn+fp+fn))
    # Outputing the time it took to perform cross validation
    print("Cross validation took %.2f seconds to complete." % (time.time()-t0))
    # The function returns the mean accuracy score, as well 
    # as the standard deviation from the mean
    return np.mean(fold_accuracy), np.std(fold_accuracy)
            
# Function that splits the training data into folds
def create_folds(k_folds, x_train, y_train):
    # Setting up the data variables that will shrink over time as the 
    # folds are extracted from the data to not repeat
    x_data, y_data = x_train, y_train
    # Setting up the array that will contain the folds
    folds = []
    # Looping the wanted amount of times
    for i in range(k_folds):
        # Calculating the percentage of data that will become the next fold
        percentage = 1/(k_folds-i)
        # If the percentage becomes 100, then the scikit train_test_split won't work as intended, 
        # so to avoid possible errors, if the percentage becomes 100% (meaning we're on the last 
        # fold) then the remaining data just gets added to the fold array. If we're not at 100%
        if percentage != 1:
            # The scikit learn train_test_slpit function is used to split up the data evenly, so that the 
            # folds all have some amount of spam as well as ham
            x_data, fold_x, y_data, fold_y = train_test_split(x_data, y_data, test_size = percentage, random_state=50)
            # after the split, the "test" data is the new fold, which is turned into a dictionary
            # and added into the folds array, while data is set to the data that wasn't used in this fold
            folds.append(create_dictionary(fold_x, fold_y))
        else: folds.append(create_dictionary(x_data, y_data))
    # The function returns the folds
    return folds

#endregion

#region -------------- running the code --------------------
# Setting the initial parameters
big, itr, lc, points_num, k_folds = set_params()
# Getting the dataset from data.py
data.count_words(big)
df = data.get_data(big)
words_ham, words_spam = data.order_words(big)
x_train, x_test, y_train, y_test = train_test_split(df['email'], df['label'], test_size = 1/3, random_state = 50)
# Converting the data from data.py into dictionaries. This choice was done mainly
# because the perceptron seemed easier to implement with the use of dictionaries.
words_ham, words_spam = dict(words_ham), dict(words_spam)
training_set = create_dictionary(x_train, y_train)
testing_set = create_dictionary(x_test, y_test)

# The weight dictionary is instanciated, 
# before being sent into the learn_weights function
weights = {}

# Running the training algorithm with the provided parameters.
lowest_threshold, highest_threshold = learn_weights(weights, training_set, itr, lc)

# The test function returns the four values ov a confusion matrix
tp, tn, fp, fn = test_weights(weights, testing_set)
print("The model performed with an accuracy of %.2f%s\n" % ((tp+tn)/(tp+tn+fp+fn), '%'))

# Plotting the confusion matrix of a perceptron with the threshold set to 0
plot_cm([[tp, fp],[fn, tn]])

# Preforming cross validation on the perceptron
acc, std = cross_validation(k_folds, itr, lc, x_train, y_train)
print("In cross validation the model performed with an accuracy of %.2f%s\nwith a standard deviation of +- %.2f\n" % (acc, '%', std))

# Plotting the ROC curve.
plot_roc(itr, lc, points_num, lowest_threshold, highest_threshold, training_set, testing_set)

#endregion