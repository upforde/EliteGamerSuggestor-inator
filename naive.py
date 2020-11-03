# Importing the data module
import data
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix

# Create a multiomail naive bayes item
# Feed the naive bayes item top used words from each classification

# Probablity of Spam
prob_spam = data.num_spam / data.num_total
print(prob_spam)

df = data.get_data()



#print(df.head)

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


print(all_features.vocabulary_)

#Use all_features and X_train like this:

model1 = MultinomialNB()
model2 = LinearSVC()
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

#I think we need a bigger dataset......
result1 = model1.predict(X_test)
result2 = model2.predict(X_test)
print("Naive Bayes")
print(confusion_matrix(y_test,result1))
print("Support Vector Machine")
print(confusion_matrix(y_test,result2))