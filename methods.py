# Importing the data + visual module, and other methods from the sklearn, seaborn, panda, numpy, and matplotlib libraries.
import data
import numpy as np
import visuals
import seaborn as sn
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# LEGACY, Never used.
# Check if object is a float or not. Some of the e-mail words are assumed to be floats. This was to be used to tackle that problem during feature extraction.
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# LEGACY, Redundant.
# Create a vocabulary of all words that exists in our dataframe. To be used in feature_extraction.
# However, a better method of word list exists in data.py.
# Param  : df           = Dataframe to create dictionary from. 
# Output : vocabulary   = Vocabulary of words and they number of occurences as dict.
def makeDict(df):
    # Make vocabulary ready for words
    vocabulary = []
    # Iterate through each tuple to grap their text content
    for tuples in df.itertuples():
        content = tuples[1]
        # Fail-safe for text content assumed to be floats by the file reader.
        if is_number(content):
            content = str(content)
        # Seperate content
        content = content.split()
        # Add content to vocabulary
        vocabulary += content
        # Create list of words and the number of the occurences.
        vocabulary = Counter(list(dict.fromkeys(vocabulary)))
    return vocabulary

# LEGACY, Redundant.
# Manual feature extraction. Lacks major benefits such as parallel processing. Opted into scikit CountVectorizer for numerous reasons.
# Not even sure this one works anymore after numerous refractors...
# Param  : smalldata     = Boolean value if feature extraction is done on the enron dataset or small dataset(kaggle).
# Param  : clean         = Cleaning list which includes bolean values to be put into the count_word function.
# Output : feature_ma..  = Feature matrix to be used for training model. A bag-of-words model that counts word occurences.
def feature_extraction(smalldata, clean):
    # Gets the dataframe and fills up word counting dictionaries. ¨
    # Also cleans data.
    df = data.count_words(smalldata, clean[0], clean[1], clean[2])
    # Get correct word dictionary for the dataset we are using.
    if smalldata:
        word_dict = data.df_dict
    else:
        word_dict = data.big_df_dict
    # Make feature matrix ready to be filled. Create it with number of instances and number of words.
    feature_matrix = np.zeros((data.num_total(df), len(word_dict)))
    # Iterate through each row.
    for tuples in df.itertuples():
        # Variable to hold all words for that instance.
        all_words = []
        # Grab text content from row.
        content = tuples[1]
        # Check if the row is considered a float, if it is, convert it into a string.
        # Should not be a problem if properly cleaned...
        if is_number(content):
            content = str(content)
        # Split up text content by whitespaces, creating words.
        content = content.split()
        all_words.extend(content)
        # For each word in all words that exists in the current instance...
        # Check for every word in vocabulary if it exists in the e-mail, if it does, count all occurences for that word index.
        for i, value in enumerate(word_dict.keys()):
            if value in all_words:
                feature_matrix[tuples[0], i] = all_words.count(value)
    return feature_matrix

# The main function for training LinearSVC-, Multinomial Naive Bayes-, and Bernoulli Naive Bayes classification models.
# Trains the model accordingly with numerous parameters to allow permutations.
# First retrieves and cleans a dataframe based on paramters, then creates a vectorizer to perform feature extraction.
# Second, if n_fold is higher than 2, performs a cross_validation on our dataset. Otherwise, a 1/3 data-split.
# Third, fits the model sent in with training data and asks it to predict the validation data for the current loop(s).
# Fourth, based on parameters, output necessary information. Confusion matrix, F1-score, etc. This is where plots are called if asked for.
# Lastly, output everything and return either the classifier alone, or the classifier and necessary data for plotting our ROC-curve.
# Note, if returning a list of data, we have to extend the returning data to an assigned list.
# Param  : model         = Classification model to be trained and used for predictions and plotting.
# Param  : smalldata     = If true, the small dataframe will be used. Otherwise big dataframe will be used.
# Param  : clean         = List of three bool values to determine if [0]cleaning, [1]lemmatization, and [2]stemming is to be done.
# Param  : vector_type   = Vectorization type to be used when performing bag-of-words for training the models.
# Param  : n_folds       = Integer that determines number of k-folds. If less than 2, performs, 1/3 test/training split.
# Param  : report        = If true, the function should output everything necessary for the project. Plots, shows, and saves everything to /images. 
# Param  : collect_roc   = If true, the function should collect and return a second output in the form of a list to be used for drawing the ROC-curve. 
# Output : model         = Returns the now trained classification model that got sent in from the first parameter.
# Output : roc_data_list = (Optimal) ROC-curve data to be further pipelined into the ROC-curve plotting function.
def train_model(model = MultinomialNB(), smalldata = True, clean = [False, False, False], vector_type = CountVectorizer(), n_folds = 0, report = False, collect_roc = False):
    print_reciept(model, smalldata, clean, vector_type, n_folds)
    print("Fetching dataframe...")
    df = data.count_words(smalldata, clean[0], clean[1], clean[2])
    if not isinstance(vector_type, (CountVectorizer, TfidfVectorizer)):
        print("Vectorizer uses invalid format for this project, cancelling run!")
        return model
    features = vector_type.fit_transform(df['email'].apply(lambda x: np.str_(x)))
    labels = df['label'].apply(lambda x: np.str_(x))
    roc_data_list = []
    if n_folds > 1:
        print(f"Performing {n_folds}-fold cross validation...")
        k_fold = KFold(n_splits = n_folds, shuffle=True) # Very interesting... Removing the shuffle will make fold not be able to predict Hams.
        confusion_matrix_total = np.array([[0, 0], [0, 0]])
        avg_score = []
        for i, (train_index, test_index) in enumerate(k_fold.split(features)):   
            print(" ")
            print(f"Prediction for fold {i}")
            print(f"================={i}==================")
            X_train = features[train_index]
            X_test = features[test_index]
            y_train = labels[train_index]
            y_test = labels[test_index]
            if isinstance(model, (MultinomialNB, BernoulliNB, LinearSVC)):
                model.fit(X_train, y_train)
                result = model.predict(X_test)
            else:
                print("Wrong model used, cancelling run!")
                return model
            confusion_matrix_total += confusion_matrix(y_test, result)
            score = f1_score(y_test, result, pos_label="0")
            avg_score.append(score)
            if report:
                print(f"Classification report for fold {i}: ")
                print (classification_report(result, y_test))
                clean.insert(0,smalldata)
                roc_data_list.append(visuals.get_ROC(model, X_test, y_test, i, clean))
                visuals.plot_cm(confusion_matrix_total, model, clean)
            elif collect_roc and report is not True:
                roc_data_list.append(visuals.get_ROC(model, X_test, y_test, i))
                print(f"Score for fold {i} spam-prediction : {score}")
                print(f"Confusion matrix for fold {i}:")
                print(confusion_matrix_total)
            else:
                print(f"Score for fold {i} spam-prediction : {score}")
                print(f"Confusion matrix for fold {i}:")
                print(confusion_matrix_total)
        if report:
            _ = visuals.draw_ROC(roc_data_list)
        print('Average score:', sum(avg_score)/len(avg_score))
    else:
        print("Performing 1/3 test-data split...")
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33)
        if isinstance(model, (MultinomialNB, BernoulliNB, LinearSVC)):
            model.fit(X_train, y_train)
            result = model.predict(X_test)
        else:
            print("Wrong model used, cancelling run!")
            return model
        conf_matrix = confusion_matrix(y_test, result)
        score = f1_score(y_test, result, pos_label="0")
        if report:
            print("Classification report: ")
            print (classification_report(result, y_test))
            clean.insert(0,smalldata)
            roc_data_list.append(visuals.get_ROC(model, X_test, y_test, None, clean))
            _ = visuals.draw_ROC(roc_data_list)
            visuals.plot_cm(conf_matrix, model, clean)
        elif collect_roc and report is not True:
            roc_data_list.append(visuals.get_ROC(model, X_test, y_test))
            print(f"Score for spam-prediction : {score}")
            print("Confusion matrix:")
            print(conf_matrix)
        else:
            print(f"Score for spam-prediction : {score}")
            print("Confusion matrix:")
            print(conf_matrix)
    if collect_roc:
        return model, roc_data_list
    else:
        return model

def print_reciept(model, smalldata, clean, vector_type, n_folds):
    print(" ")
    print(" ")
    print(" ")
    if smalldata:
        datainfo = "the kaggle e-mail dataset (small dataset)."
    else:
        datainfo = "the enron e-mail dataset (big dataset)."
    print(f"Traning a \033[1m{model.__class__.__name__}\033[0m model with the \033[1m{datainfo}\033[0m.")
    print(f"Using \033[1m{vector_type.__class__.__name__}\033[0m-type of bag-of-words approach.")
    if clean[0]:
        print("Dataframe will be \033[1m cleaned\033[0m. for stop-, and exclusion words.")
        if clean[1] and clean[2]:
            print("Dataframe will be also go through \033[1mlemmitization and stemming.\033[0m")
        elif clean[1]:
            print("Dataframe will be also go through \033[1mlemmitization.\033[0m")
        else:
            print("Dataframe will be also go through \033[1mstemming.\033[0m")
    else:
        print("Dataframe will \033[1m not be cleaned\033[0m.")
    if n_folds > 1:
        print(f"\033[1m{n_folds}-fold cross validation\033[0m will be used.")
    else:
        print("\033[1m 1/3 test-data split\033[0m will be used.")
    print("#=============================================#")



# These are just to give an idea on how to use the function.
# Permutations are, however, commented out at the bottom if one would like to try and run them. :)

print("Small data-set example.")
print("Cleaning with stemming.")
clean_param = [True, False, True]
roc_list = []
model_mnb = train_model(MultinomialNB(), True, clean_param, CountVectorizer(), 0, True)
model_bnb, roc_list[len(roc_list):] = train_model(BernoulliNB(), True, clean_param, CountVectorizer(binary=True), 0, True, True)
model_svm, roc_list[len(roc_list):] = train_model(LinearSVC(max_iter=10000), True, clean_param, CountVectorizer(), 0, True, True)
model_mnb_4fold, roc_list[len(roc_list):] = train_model(MultinomialNB(), True, clean_param, CountVectorizer(), 4, False, True)
roc_list = visuals.draw_ROC(roc_list)

print("Big data-set example.")
print("Cleaning with Lemmatization...")

model_mnb, roc_list[len(roc_list):] = train_model(MultinomialNB(), False, clean_param, CountVectorizer(), 0, True, True)
model_bnb, roc_list[len(roc_list):] = train_model(BernoulliNB(), False, clean_param, CountVectorizer(binary=True), 0, True, True)



"""
print("Small data-set permutations.")
print("No cleaning.")

clean_param = [False, False, False]
roc_list = []

model_mnb, roc_list[len(roc_list):] = train_model(MultinomialNB(), True, clean_param, CountVectorizer(), 0, True, True)
model_bnb, roc_list[len(roc_list):] = train_model(BernoulliNB(), True, clean_param, CountVectorizer(binary=True), 0, True, True)
model_svm, roc_list[len(roc_list):] = train_model(LinearSVC(max_iter=10000), True, clean_param, CountVectorizer(), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)

print("Lemmitazation, no stemming.")
clean_param = [True, True, False]

model_mnb, roc_list[len(roc_list):] = train_model(MultinomialNB(), True, clean_param, CountVectorizer(), 0, True, True)
model_bnb, roc_list[len(roc_list):] = train_model(BernoulliNB(), True, clean_param, CountVectorizer(binary=True), 0, True, True)
model_svm, roc_list[len(roc_list):] = train_model(LinearSVC(max_iter=10000), True, clean_param, CountVectorizer(), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)

print("No lemmitazation, stemming.")
clean_param = [True, False, True]

model_mnb, roc_list[len(roc_list):] = train_model(MultinomialNB(), True, clean_param, CountVectorizer(), 0, True, True)
model_bnb, roc_list[len(roc_list):] = train_model(BernoulliNB(), True, clean_param, CountVectorizer(binary=True), 0, True, True)
model_svm, roc_list[len(roc_list):] = train_model(LinearSVC(max_iter=10000), True, clean_param, CountVectorizer(), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)

print("Cleaning with lemmatization and stemming.")
clean_param = [True, True, True]

model_mnb, roc_list[len(roc_list):] = train_model(MultinomialNB(), True, clean_param, CountVectorizer(), 0, True, True)
model_bnb, roc_list[len(roc_list):] = train_model(BernoulliNB(), True, clean_param, CountVectorizer(binary=True), 0, True, True)
model_svm, roc_list[len(roc_list):] = train_model(LinearSVC(max_iter=10000), True, clean_param, CountVectorizer(), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)


# ========================================


print("Big data-set permutations.")

print("No cleaning.")

clean_param = [False, False, False]
roc_list = []

model_mnb, roc_list[len(roc_list):] = train_model(MultinomialNB(), False, clean_param, CountVectorizer(), 0, True, True)
model_bnb, roc_list[len(roc_list):] = train_model(BernoulliNB(), False, clean_param, CountVectorizer(binary=True), 0, True, True)
model_svm, roc_list[len(roc_list):] = train_model(LinearSVC(max_iter=100000), False, clean_param, CountVectorizer(), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)

print("Lemmitazation, not stemming.")
clean_param = [True, True, False]

model_mnb, roc_list[len(roc_list):] = train_model(MultinomialNB(), False, clean_param, CountVectorizer(), 0, True, True)
model_bnb, roc_list[len(roc_list):] = train_model(BernoulliNB(), False, clean_param, CountVectorizer(binary=True), 0, True, True)
model_svm, roc_list[len(roc_list):] = train_model(LinearSVC(max_iter=100000), False, clean_param, CountVectorizer(), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)

print("No lemmitazation, stemming.")
clean_param = [True, False, True]

model_mnb, roc_list[len(roc_list):] = train_model(MultinomialNB(), False, clean_param, CountVectorizer(), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)
model_bnb, roc_list[len(roc_list):] = train_model(BernoulliNB(), False, clean_param, CountVectorizer(binary=True), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)
model_svm, roc_list[len(roc_list):] = train_model(LinearSVC(max_iter=100000), False, clean_param, CountVectorizer(), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)

print("Cleaning with lemmatization and stemming.")
clean_param = [True, True, True]

model_mnb, roc_list[len(roc_list):] = train_model(MultinomialNB(), False, clean_param, CountVectorizer(), 0, True, True)
model_bnb, roc_list[len(roc_list):] = train_model(BernoulliNB(), False, clean_param, CountVectorizer(binary=True), 0, True, True)
model_svm, roc_list[len(roc_list):] = train_model(LinearSVC(max_iter=100000), False, clean_param, CountVectorizer(), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)
"""