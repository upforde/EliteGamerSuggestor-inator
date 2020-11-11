import methods as md
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib.colors import ListedColormap

# Compute ROC curve and ROC area for each method used
# Baseline.
r_prob = [0 for _ in range(len(md.y_test))]

# Get the probability of the predictions.
mnnb_prob = md.modelMNB.predict_proba(md.X_test)
bnb_prob = md.modelBNB.predict_proba(md.X_test)
gnnb_prob = md.modelGNB.predict_proba(md.X_test.todense())
svm_prob = md.modelSVM.decision_function(md.X_test)

# Keep positive outcome of the methods
mnnb_prob = mnnb_prob[:, 1]
bnb_prob = bnb_prob[:, 1]
gnnb_prob = gnnb_prob[:, 1]
#svm_prob = svm_prob[:, 1]

#print(mnnb_prob.shape)
#print(gnnb_prob.shape)

# Compute the AUROC, area under receiver operating characteristic
r_auc = roc_auc_score(md.y_test, r_prob)
mnnb_auc = roc_auc_score(md.y_test, mnnb_prob)
bnb_auc = roc_auc_score(md.y_test, bnb_prob)
gnnb_auc = roc_auc_score(md.y_test, gnnb_prob)
svm_auc = roc_auc_score(md.y_test, svm_prob)


#print(r_auc)
#print(mnnb_auc)
#print(gnnb_auc)

md.y_test= '1' <= md.y_test

# Get false positives and true positives to be plotted on the ROC curve
r_fpr, r_tpr, _ = roc_curve(md.y_test, r_prob)
mnnb_fpr, mnnb_tpr, mnnb_threshold = roc_curve(md.y_test, mnnb_prob)
bnb_fpr, bnb_tpr, bnb_threshold = roc_curve(md.y_test, bnb_prob)
gnnb_fpr, gnnb_tpr, gnnb_threshold = roc_curve(md.y_test, gnnb_prob)
svm_fpr, svm_tpr, svm_threshold = roc_curve(md.y_test, svm_prob)

"""
print(gnnb_fpr)
print(gnnb_tpr)
print(mnnb_fpr)
print(mnnb_tpr)
print(mnnb_threshold.shape)
print(bnb_threshold.shape)
print(gnnb_threshold.shape)
print(svm_threshold.shape)
"""

# Plotting the ROC-curve with the models we've built
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

# Plotting the decision boundary with the models we've built
xlim = (-1, 8)
ylim = (-1, 5)
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 71),
                     np.linspace(ylim[0], ylim[1], 81))
Z = md.modelGNB.predict_proba(np.c_[xx.ravel(), yy.ravel()])
(2010, 33813) (2010,)
Z = Z[:, 1].reshape(xx.shape)



"""
# Defining the boundaries
x_min = md.all_features[:, 0].min()-1
x_max = md.all_features[:, 0].max()+1
y_min = md.all_features[:, 1].min()-1
y_max = md.all_features[:, 1].max()+1

xgrid = np.arange(x_min, x_max, 0.1)
ygrid = np.arange(y_min, y_max, 0.1)

# Defining the X and Y scale
xx, yy = np.meshgrid(xgrid, ygrid)

print(xx.ravel().shape)
print(yy.ravel().shape)

Z = md.modelGNB.predict(np.c_[xx.ravel(), yy.ravel()])

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(md.X_train[:, 0], md.X_train[:, 1], c=md.y_train, cmap=cmap_bold)

y_predicted = md.modelGNB.predict(md.X_test)
score = md.modelGNB.score(md.X_test, md.y_test)
plt.scatter(md.X_test[:, 0], md.X_test[:, 1], c=y_predicted, alpha=0.5, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
"""