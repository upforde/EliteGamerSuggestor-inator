import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.metrics import roc_auc_score, roc_curve

def get_ROC(model, X_test, y_test, i = None, clean = None):
    # Get the probability of the predictions.
    # Keep positive outcome of the methods if function abides it.
    if isinstance(model, LinearSVC):
        prob = model.decision_function(X_test)
    elif isinstance(model, GaussianNB):
        # Apparently the probabilities are very close to zero or one. Forced to use log on this one.
        prob = model.predict_log_proba(X_test.toarray())
        prob = prob[:, 1]
        print(prob)
    else:
        prob = model.predict_proba(X_test)
        prob = prob[:, 1]
    # Compute the AUROC, area under receiver operating characteristic
    auc = roc_auc_score(y_test, prob)
    y_test= '1' <= y_test
    # Get false positives and true positives to be plotted on the ROC curve
    fpr, tpr, threshold = roc_curve(y_test, prob)
    # Return values to be used in drawing
    label = get_label(model, i)
    if clean is not None:
        return [fpr, tpr, threshold, auc, label, clean]
    else:
        return [fpr, tpr, threshold, auc, label]

# Plotting the ROC-curve with the models we've built
def draw_ROC(roc_list):
    filename = None
    for roc_data in roc_list:
        plt.plot(roc_data[0], roc_data[1], marker='.', label=f" {roc_data[4]} with {len(roc_data[2])} points : AUC = {round(roc_data[3],2)}")
        if len(roc_data) == 6:
            dataname = get_clean_name(roc_data[5])
            filename = roc_data[4] + dataname
    plt.plot(plt.xlim(), plt.ylim(), linestyle="--")
    plt.title("ROC plot")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.legend()
    if filename is not None:
        plt.savefig(f"images/ROC{filename}.png")
    else:
        plt.show()
    plt.clf()
    roc_list = []
    return roc_list

def get_label(model, i):
    if i is None:
        label = model.__class__.__name__
    else:
        label = f"{model.__class__.__name__} with fold {i}"
    return label

def plot_cm(cm, model, clean, i = None):
    filename = get_label(model, i) + get_clean_name(clean)
    df_cm = pd.DataFrame(cm, index = ["Real Ham", "Real Spam"], columns = ["Guessed Ham", "Guessed Spam"])
    sn.heatmap(df_cm, annot=True, fmt="d", cmap=sn.color_palette("rocket_r", as_cmap=True))
    plt.savefig(f"images/Matrix{filename}.png")
    plt.clf()

def get_clean_name(clean):
    cleandata = ""
    filename = ""
    if clean[0]:
        if clean[1] and clean[2]:
            cleandata = "Lemm And Stem"
        elif clean[1]:
            cleandata = " Lemm"
        else:
            cleandata = " Stem"
    else:
        filename = "Not Clean"
    return filename + cleandata
