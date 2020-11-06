# Importing the data module
import data
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Probablity of Spam
prob_spam = data.num_spam / data.num_total
print(prob_spam)

dfcleaned = data.get_data()

df = dfcleaned.copy()

data.clean_data()


# Removes stop words
vectorizer = CountVectorizer(stop_words='english')

# Create a vectorization from all the content of email coloumn
all_features = vectorizer.fit_transform(df['email'].apply(lambda x: np.str_(x)))
all_features_cleaned = vectorizer.fit_transform(dfcleaned['email'].apply(lambda x: np.str_(x)))

# Get possible labels
labels = df['label'].apply(lambda x: np.str_(x))

print(all_features)
#Separates mail content and indexes it with its id for email, 3000

separated = df['email']
print(separated)
print(type(separated))


"""
for word in str(email).split():
            if word in words_dict:
                words_dict[word]= words_dict[word] + 1
            else:
                words_dict[word] = 1
"""

def feature_extraction(df):
    data.count_words(False, False, False)
    feature_matrix = np.zeros((data.num_total, len(data.words_ham) + len(data.words_ham)))
    all_words = []
    for index, row in df.iterrows():
        for content in row['email']:
            words = content.split()
            all_words += words
            for word in all_words:
                for wordIndex in range(len(all_words)):
                    feature_matrix[index,wordIndex] = all_words.count(word)
    return feature_matrix

print(feature_extraction(df))

"""
# Split training data
X_train, X_test, y_train, y_test = train_test_split(all_features, labels, test_size=0.33, random_state=50)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(all_features_cleaned, labels, test_size=0.33, random_state=50)

# Amount of training data, test/validation data, and combined e-mails and counted words
print("Original data features")
print(X_train.shape, X_test.shape, all_features.shape)
print("Cleaned data features")
print(X_train_c.shape, X_test_c.shape, all_features_cleaned.shape)

# Instantiate methods through scikit
modelMNB = MultinomialNB()
modelSVM = LinearSVC(max_iter=10000)
modelBNB = BernoulliNB(alpha = 1.0)
modelGNB = GaussianNB()

# Instantiate methods through scikit for cleaned data
modelMNBc = MultinomialNB()
modelSVMc = LinearSVC(max_iter=10000)
modelBNBc = BernoulliNB(alpha = 1.0)
modelGNBc = GaussianNB()

# Fit the model using our features.
modelMNB.fit(X_train, y_train)
modelSVM.fit(X_train, y_train)
modelBNB.fit(X_train, y_train)
modelGNB.fit(X_train.todense(), y_train)

# Fit the models using cleaned data.
modelMNBc.fit(X_train_c, y_train_c)
modelSVMc.fit(X_train_c, y_train_c)
modelBNBc.fit(X_train_c, y_train_c)
modelGNBc.fit(X_train_c.todense(), y_train_c)


# Predict classification using our test data
resultMNB = modelMNB.predict(X_test)
resultSVM = modelSVM.predict(X_test)
resultBNB = modelBNB.predict(X_test)
resultGNB = modelGNB.predict(X_test.todense())

# Predict classification using our cleaned test data
resultMNBc = modelMNBc.predict(X_test_c)
resultSVMc = modelSVMc.predict(X_test_c)
resultBNBc = modelBNBc.predict(X_test_c)
resultGNBc = modelGNBc.predict(X_test_c.todense())

# Print out confusion matrix and classification report. <-- More on this later.
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

print(" ")
print("Multinomial Naive Bayes")
print (classification_report(resultMNB, y_test))

print(" ")
print("Bernoulli Naive Bayes")
print (classification_report(resultBNB, y_test))

print(" ")
print("Gaussian Naive Bayes")
print (classification_report(resultGNB, y_test))
#https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur

print(" ")
print("Support Vector Machine")
print (classification_report(resultSVM, y_test))

print(" ")
print(" ")
print("Cleaned data classification performance")
print(" ")
print(" ")

print(" ")
print("Multinomial Naive Bayes")
print (classification_report(resultMNB, y_test_c))

print(" ")
print("Bernoulli Naive Bayes")
print (classification_report(resultBNBc, y_test_c))

print(" ")
print("Gaussian Naive Bayes")
print (classification_report(resultGNBc, y_test_c))
#https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur

print(" ")
print("Support Vector Machine")
print (classification_report(resultSVMc, y_test_c))


# Compute ROC curve and ROC area for each method used
# Baseline
r_prob = [0 for _ in range(len(y_test))]

# Get the probability of the predictions
mnnb_prob = modelMNB.predict_proba(X_test)
bnb_prob = modelBNB.predict_proba(X_test)
gnnb_prob = modelGNB.predict_proba(X_test.todense())
svm_prob = modelSVM.decision_function(X_test)

# Keep positive outcome of the methods
mnnb_prob = mnnb_prob[:, 1]
bnb_prob = bnb_prob[:, 1]
gnnb_prob = gnnb_prob[:, 1]
#svm_prob = svm_prob[:, 1]

print(mnnb_prob.shape)
print(gnnb_prob.shape)

# Compute the AUROC, area under receiver operating characteristic
r_auc = roc_auc_score(y_test, r_prob)
mnnb_auc = roc_auc_score(y_test, mnnb_prob)
bnb_auc = roc_auc_score(y_test, bnb_prob)
gnnb_auc = roc_auc_score(y_test, gnnb_prob)
svm_auc = roc_auc_score(y_test, svm_prob)


#print(r_auc)
print(mnnb_auc)
print(gnnb_auc)

y_test= '1' <= y_test

# Get false positives and true positives to be plotted on the ROC curve
r_fpr, r_tpr, _ = roc_curve(y_test, r_prob)
mnnb_fpr, mnnb_tpr, mnnb_threshold = roc_curve(y_test, mnnb_prob)
bnb_fpr, bnb_tpr, bnb_threshold = roc_curve(y_test, bnb_prob)
gnnb_fpr, gnnb_tpr, gnnb_threshold = roc_curve(y_test, gnnb_prob)
svm_fpr, svm_tpr, svm_threshold = roc_curve(y_test, svm_prob)

print(gnnb_fpr)
print(gnnb_tpr)
print(mnnb_fpr)
print(mnnb_tpr)
print(mnnb_threshold.shape)
print(bnb_threshold.shape)
print(gnnb_threshold.shape)
print(svm_threshold.shape)


plt.plot(r_fpr, r_tpr, linestyle="--")
plt.plot(mnnb_fpr, mnnb_tpr, marker='.', label="Multinomial Naive Bayes, " % mnnb_auc)
plt.plot(bnb_fpr, bnb_tpr, marker='.', label="Bernoulli Naive Bayes, " % mnnb_auc)
plt.plot(gnnb_fpr, gnnb_tpr, marker='.', label="Gaussian Naive Bayes, " % mnnb_auc)
plt.plot(svm_fpr, svm_tpr, marker='.', label="Support Vector Machine, " % mnnb_auc)
plt.title("ROC plot")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.legend()
plt.show()
"""