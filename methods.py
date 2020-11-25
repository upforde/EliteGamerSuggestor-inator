# Importing the data module and other methods from sklearn library.
import data
import numpy as np
import visuals
import seaborn as sn
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
# Create a dictionary of all words that exists in our dataframe.
# Param: df = Dataframe to create dictionary from. 
def makeDict(df):
    # Make dictionary ready for words
    dictionary = []
    # Iterate through each tuple to grap their text content
    for tuples in df.itertuples():
        content = tuples[1]
        # Fail-safe for text content assumed to be floats by the file reader.
        if is_number(content):
            content = str(content)
        # Seperate content
        content = content.split()
        # Add content to dictonary
        dictionary += content
        # Create sorted list with highest occuring word at the top.
        dictionary = list(dict.fromkeys(dictionary))
    return dictionary

# LEGACY, Redundant.
# Manual feature extraction. Lacks major benefits such as parallel processing. Opted into scikit CountVectorizer for numerous reasons.
# Not even sure this one works anymore after numerous refractors...
# Param : smalldata     = Boolean value if feature extraction is done on the enron dataset or small dataset(kaggle).
# Param : clean         = Cleaning list which includes bolean values to be put into the count_word function. 
def feature_extraction(smalldata, clean):
    df = data.count_words(smalldata, clean[0], clean[1], clean[2])
    if smalldata:
        word_dict = data.df_dict
    else:
        word_dict = data.big_df_dict
    feature_matrix = np.zeros((data.num_total(df), len(word_dict)))
    for tuples in df.itertuples():
        all_words = []
        content = tuples[1]
        if is_number(content):
            content = str(content)
        content = content.split()
        all_words.extend(content)
        for word in all_words:
            for counter, value in enumerate(word_dict.keys()):
                if value == word:
                    feature_matrix[tuples[0], counter] = all_words.count(word)
    return feature_matrix

# The main training function for multiple models.
# Trains the model accordingly with numerous parameters to allow permutations.
# Param : model         = Classification model to be trained and used for predictions and plotting.
# Param : smalldata     = Bolean value to decide if the small dataframe or big dataframe is to be used.
# Param : clean         = List of three bolean values to determine if [0]cleaning, [1]lemmatization, and [2]stemming is to be done.
# Param : vector_type   = Vectorization type to be used when performing bag-of-words for training the models.
# Param : n_folds       = Integer that determines number of k-folds. If less than 2, performs, 1/3 test/training split.
# Param : report        = Bolean value to determine if a full classification report will be output.
# Param : collect_roc   = Bolean value to determine if the function should collect and return data to be used for drawing ROC-curve.
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
        for i, (train_index, test_index) in enumerate(k_fold.split(features)):
            X_train = features[train_index]
            X_test = features[test_index]
            y_train = labels[train_index]
            y_test = labels[test_index]

            print(" ")
            print(f"Prediction for fold {i}")
            print(f"================={i}==================")

            if isinstance(model, (MultinomialNB, BernoulliNB, LinearSVC)):
                model.fit(X_train, y_train)
                result = model.predict(X_test)

            elif isinstance(model, GaussianNB):
                model.fit(X_train.toarray(), y_train)
                result = model.predict(X_test.toarray())
            else:
                print("Wrong model used!")
            confusion_matrix_total += confusion_matrix(y_test, result)
            score = f1_score(y_test, result, pos_label="0")
            print(f"Confusion matrix for {i}:")
            print(confusion_matrix_total)
            if report:
                print(f"Classification report for {i}: ")
                print (classification_report(result, y_test))
                visuals.plot_cm(confusion_matrix_total, model, clean)
            else:
                print(f"Score for {i} spam-prediction : {score}")
            if collect_roc and report:
                roc_data_list.append(visuals.get_ROC(model, X_test, y_test, i, clean))
            elif collect_roc:
                roc_data_list.append(visuals.get_ROC(model, X_test, y_test, i))
    else:
        print("Performing 1/3 test-data split...")
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33)
        if isinstance(model, (MultinomialNB, BernoulliNB, LinearSVC)):
            model.fit(X_train, y_train)
            result = model.predict(X_test)
        elif isinstance(model, GaussianNB):
            model.fit(X_train.toarray(), y_train)
            result = model.predict(X_test.toarray())
        else:
            print("Wrong model used!")
        conf_matrix = confusion_matrix(y_test, result)
        score = f1_score(y_test, result, pos_label="0")
        print("Confusion matrix:")
        print(conf_matrix)
        if report:
            print("Classification report for: ")
            print (classification_report(result, y_test))
            visuals.plot_cm(conf_matrix, model, clean)
        else:
            print(f"Score for spam-prediction : {score}")
        if collect_roc and report:
            roc_data_list.append(visuals.get_ROC(model, X_test, y_test, None, clean))
        elif collect_roc and report is not True:
            roc_data_list.append(visuals.get_ROC(model, X_test, y_test))
    if collect_roc:
        return model, roc_data_list
    else:
        return model

def print_reciept(model, smalldata, clean, vector_type, n_folds):
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
"""
print("Small data-set permutations.")
print("No cleaning.")

clean_param = [False, False, False]
roc_list = []

model_mnb, roc_list[len(roc_list):] = train_model(MultinomialNB(), True, clean_param, CountVectorizer(), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)
model_bnb, roc_list[len(roc_list):] = train_model(BernoulliNB(), True, clean_param, CountVectorizer(binary=True), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)
model_svm, roc_list[len(roc_list):] = train_model(LinearSVC(max_iter=10000), True, clean_param, CountVectorizer(), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)

print("Lemmitazation, not stemming.")
clean_param = [True, True, False]

model_mnb, roc_list[len(roc_list):] = train_model(MultinomialNB(), True, clean_param, CountVectorizer(), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)
model_bnb, roc_list[len(roc_list):] = train_model(BernoulliNB(), True, clean_param, CountVectorizer(binary=True), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)
model_svm, roc_list[len(roc_list):] = train_model(LinearSVC(max_iter=10000), True, clean_param, CountVectorizer(), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)

print("No lemmitazation, stemming.")
clean_param = [True, False, True]

model_mnb, roc_list[len(roc_list):] = train_model(MultinomialNB(), True, clean_param, CountVectorizer(), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)
model_bnb, roc_list[len(roc_list):] = train_model(BernoulliNB(), True, clean_param, CountVectorizer(binary=True), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)
model_svm, roc_list[len(roc_list):] = train_model(LinearSVC(max_iter=10000), True, clean_param, CountVectorizer(), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)

print("No lemmitazation, stemming.")
clean_param = [True, True, True]

model_mnb, roc_list[len(roc_list):] = train_model(MultinomialNB(), True, clean_param, CountVectorizer(), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)
model_bnb, roc_list[len(roc_list):] = train_model(BernoulliNB(), True, clean_param, CountVectorizer(binary=True), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)
model_svm, roc_list[len(roc_list):] = train_model(LinearSVC(max_iter=10000), True, clean_param, CountVectorizer(), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)

# ========================================
"""
print("Big data-set permutations.")
print("No cleaning.")

clean_param = [False, False, False]
roc_list = []

model_mnb, roc_list[len(roc_list):] = train_model(MultinomialNB(), False, clean_param, CountVectorizer(), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)
model_bnb, roc_list[len(roc_list):] = train_model(BernoulliNB(), False, clean_param, CountVectorizer(binary=True), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)
model_svm, roc_list[len(roc_list):] = train_model(LinearSVC(max_iter=100000), False, clean_param, CountVectorizer(), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)

print("Lemmitazation, not stemming.")
clean_param = [True, True, False]

model_mnb, roc_list[len(roc_list):] = train_model(MultinomialNB(), False, clean_param, CountVectorizer(), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)
model_bnb, roc_list[len(roc_list):] = train_model(BernoulliNB(), False, clean_param, CountVectorizer(binary=True), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)
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

print("No lemmitazation, stemming.")
clean_param = [True, True, True]

model_mnb, roc_list[len(roc_list):] = train_model(MultinomialNB(), False, clean_param, CountVectorizer(), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)
model_bnb, roc_list[len(roc_list):] = train_model(BernoulliNB(), False, clean_param, CountVectorizer(binary=True), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)
model_svm, roc_list[len(roc_list):] = train_model(LinearSVC(max_iter=100000), False, clean_param, CountVectorizer(), 0, True, True)
roc_list = visuals.draw_ROC(roc_list)





"""
model_gnb, roc_list[len(roc_list):] = train_model(GaussianNB(), False, False, CountVectorizer(), 0, True, True)
model_svm = train_model(LinearSVC(max_iter=10000), False, False, CountVectorizer(), True, False)
visuals.draw_ROC(roc_list)
roc_list = []
model_mnb, roc_list[len(roc_list):] = train_model(MultinomialNB(), False, False, CountVectorizer(), 4, False, True)
visuals.draw_ROC(roc_list)

model_mnb2 = train_model(MultinomialNB(), False, False, CountVectorizer(), 0)
model_bnb2 = train_model(BernoulliNB(), False, False, CountVectorizer(binary=True), 0)
model_gnb2 = train_model(GaussianNB(), False, False, CountVectorizer(), 0)
model_svm2 = train_model(LinearSVC(max_iter=10000), False, False, CountVectorizer(), 0)
model_mnb3 = train_model(MultinomialNB(), True, True, CountVectorizer(), 0)
model_bnb3 = train_model(BernoulliNB(), True, False, CountVectorizer(binary=True), 0)
model_gnb3 = train_model(GaussianNB(), True, False, CountVectorizer(), 0)
model_svm3 = train_model(LinearSVC(max_iter=100000), True, False, CountVectorizer(), 0)
"""