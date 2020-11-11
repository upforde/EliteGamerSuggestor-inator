# Importing the data module
import data
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report

# Probablity of Spam
# prob_spam = data.num_spam / data.num_total
# print(prob_spam)

# Fetch dataframes.
dfcleaned = data.get_data()

# Copy original dataframe before lemmatization and stemming.
df = dfcleaned.copy()

data.clean_data()


# Create a vectorizer and remove stop words
vectorizer = CountVectorizer(stop_words='english')

# Fit counting vectorizer to the content of the "email" coloumn, creating a numerical vocabulary representation of our features.
# In this case, our features are the words which appear in the content of each e-mail.
all_features = vectorizer.fit_transform(df['email'].apply(lambda x: np.str_(x)))
all_features_cleaned = vectorizer.fit_transform(dfcleaned['email'].apply(lambda x: np.str_(x)))

# Get possible labels/classifications. 0 = Spam, 1 = Ham
labels = df['label'].apply(lambda x: np.str_(x))

# LEGACY : Never used.
# Check if object is a float or not. Some of the e-mail words are assumed to be floats. This was to be used to tackle the problem during feature extraction.
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# LEGACY : Redundant.
# Create a dictionary of all words that exists in our dataframe. 
def makeDict(df):
    dictionary = []
    for tuples in df.itertuples():
        content = tuples[1]
        if is_number(content):
            content = str(content)
        content = content.split()
        dictionary += content
        dictionary = list(dict.fromkeys(dictionary))
    return dictionary

# LEGACY : Redundant.
# Manual feature extraction. Lacks major benefits such as parallel processing. Opted into scikit CountVectorizer for numerous reasons.
# Not even sure this one works...
def feature_extraction(df):
    data.count_words(True, False, False)
    feature_matrix = np.zeros((data.num_total, len(data.words)))
    for tuples in df.itertuples():
        all_words = []
        content = tuples[1]
        if is_number(content):
            content = str(content)
        content = content.split()
        all_words.extend(content)
        for word in all_words:
            for counter, value in enumerate(data.words.keys()):
                if value == word:
                    feature_matrix[tuples[0], counter] = all_words.count(word)
    return feature_matrix


# Split training data and cleaned training data.
X_train, X_test, y_train, y_test = train_test_split(all_features, labels, test_size=0.33, random_state=50)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(all_features_cleaned, labels, test_size=0.33, random_state=50)


# Amount of training data, test/validation data, and combined e-mails and counted words
print("Original data features")
print(X_train.shape, X_test.shape, all_features.shape)
print("Cleaned data features")
print(X_train_c.shape, X_test_c.shape, all_features_cleaned.shape)

# Instantiate methods to be used.
modelMNB = MultinomialNB()
modelSVM = LinearSVC(max_iter=10000)
modelBNB = BernoulliNB(alpha = 1.0)
modelGNB = GaussianNB()

# Instantiate methods for cleaned data.
modelMNBc = MultinomialNB()
modelSVMc = LinearSVC(max_iter=10000)
modelBNBc = BernoulliNB(alpha = 1.0)
modelGNBc = GaussianNB()

# Fit the model using our data.
modelMNB.fit(X_train, y_train)
modelSVM.fit(X_train, y_train)
modelBNB.fit(X_train, y_train)
modelGNB.fit(X_train.todense(), y_train)
# Fit the models using cleaned data.
modelMNBc.fit(X_train_c, y_train_c)
modelSVMc.fit(X_train_c, y_train_c)
modelBNBc.fit(X_train_c, y_train_c)
modelGNBc.fit(X_train_c.todense(), y_train_c)


# Predict classification using our test data.
resultMNB = modelMNB.predict(X_test)
resultSVM = modelSVM.predict(X_test)
resultBNB = modelBNB.predict(X_test)
resultGNB = modelGNB.predict(X_test.todense())

# Predict classification using our cleaned test data.
resultMNBc = modelMNBc.predict(X_test_c)
resultSVMc = modelSVMc.predict(X_test_c)
resultBNBc = modelBNBc.predict(X_test_c)
resultGNBc = modelGNBc.predict(X_test_c.todense())

# Print out confusion matrix and classification report.
print(" ")
print("Confusion Matrix Legend:")
print("True Positive " + "," + "False Positive ")
print("False Negative " + "," + "True Negative ")
print(" ")

print(" ")
print("Original data confusion matrix")
print(" ")

print("Multinomial Naive Bayes")
print(confusion_matrix(y_test,resultMNB))


print("Bernoulli Naive Bayes")
print(confusion_matrix(y_test,resultBNB))

print("Gaussian Naive Bayes")
print(confusion_matrix(y_test,resultGNB))

print("Support Vector Machine")
print(confusion_matrix(y_test,resultSVM))


print(" ")
print(" ")
print("Cleaned data confusion matrix")
print(" ")
print(" ")


print("Multinomial Naive Bayes")
print(confusion_matrix(y_test_c,resultMNB))

print("Bernoulli Naive Bayes")
print(confusion_matrix(y_test_c,resultBNBc))

print("Gaussian Naive Bayes")
print(confusion_matrix(y_test_c,resultGNBc))

print("Support Vector Machine")
print(confusion_matrix(y_test_c,resultSVMc))

print(" ")
print("Original data classification performance")
print(" ")

# For better labeling, as the dataset only contains 0 or 1 to address if it is a spam or not...
target_names = ["Spam", "Ham"]

print(" ")
print("Multinomial Naive Bayes")
print (classification_report(resultMNB, y_test, target_names=target_names))

print(" ")
print("Bernoulli Naive Bayes")
print (classification_report(resultBNB, y_test, target_names=target_names))

print(" ")
print("Gaussian Naive Bayes")
print (classification_report(resultGNB, y_test, target_names=target_names))

print(" ")
print("Support Vector Machine")
print (classification_report(resultSVM, y_test, target_names=target_names))

print(" ")
print(" ")
print("Cleaned data classification performance")
print(" ")
print(" ")

print(" ")
print("Multinomial Naive Bayes")
print (classification_report(resultMNB, y_test_c, target_names=target_names))

print(" ")
print("Bernoulli Naive Bayes")
print (classification_report(resultBNBc, y_test_c, target_names=target_names))

print(" ")
print("Gaussian Naive Bayes")
print (classification_report(resultGNBc, y_test_c, target_names=target_names))

print(" ")
print("Support Vector Machine")
print (classification_report(resultSVMc, y_test_c, target_names=target_names))

