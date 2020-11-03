# Importing the data module
import data
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Create a multiomail naive bayes item
# Feed the naive bayes item top used words from each classification

# Probablity of Spam
prob_spam = data.num_spam / data.num_total
print(prob_spam)

df = data.get_data()

#Removes stop words
vectorizer = CountVectorizer(stop_words='english')

#Create a vectorization from all the content of email coloumn
all_features = vectorizer.fit_transform(df['email'].apply(lambda x: np.str_(x)))

#Get possible labels
labels = df['label'].apply(lambda x: np.str_(x))

#useless
#feature_matrix = np.zeros((data.num_total, len(data.words_ham) + len(data.words_ham)))

#Split training data
X_train, X_test, y_train, y_test = train_test_split(all_features, labels, test_size=0.33, random_state=50)

#Amount of training data, test/validation data, and combined e-mails and counted words
print(X_train.shape, X_test.shape, all_features.shape)

#Instantiate methods through scikit
modelMNB = MultinomialNB()
modelSVM = LinearSVC(max_iter=10000)
modelBNB = BernoulliNB(alpha = 1.0)
modelGNB = GaussianNB()

#Fit the model using our features.
modelMNB.fit(X_train, y_train)
modelSVM.fit(X_train, y_train)
modelBNB.fit(X_train, y_train)
modelGNB.fit(X_train.todense(), y_train)

#Predict classification using our test data
result1 = modelMNB.predict(X_test)
result2 = modelSVM.predict(X_test)
result3 = modelBNB.predict(X_test)
result4 = modelGNB.predict(X_test.todense())

#Print out confusion matrix and classification report. <-- More on this later.
print(" ")
print("Confusion Matrix Legend:")
print("True Positive " + "," + "False Positive ")
print("False Negative " + "," + "True Negative ")
print(" ")

print("Multinomial Naive Bayes")
print(confusion_matrix(y_test,result1))
print (classification_report(result1, y_test))

print("Bernoulli Naive Bayes")
print(confusion_matrix(y_test,result3))
print (classification_report(result3, y_test))

print("Gaussian Naive Bayes")
print(confusion_matrix(y_test,result4))
print (classification_report(result4, y_test))
#https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur

print("Support Vector Machine")
print(confusion_matrix(y_test,result2))
print (classification_report(result2, y_test))