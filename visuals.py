import data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, roc_curve


# Function which calculates and returns the values necessary to plot the ROC-curve for a model.
# Param  : model         = The classification model used. Uses the model to create prediction probabilities for each classification.
# Param  : X_test        = The test-data set which the model uses for validation. Needed for probability predictions.
# Param  : y_test        = The test-data set which the model uses for validation. Needed for probability predictions.
# Param  : clean         = (Optimal) The cleaning parameters the model used on its dataset. Pipelined to be used for labeling of filename. If this exists, all optimal values are returned.
# Param  : smalldata     = (Optimal) Bool variable to know which dataset is used. Pipelined to be used for labeling of filename.
# Param  : vector_type   = (Optimal) The vectorization form the model used for its bag-of-words approach. Pipelined to be used for labeling of filename.
# Param  : i             = (Optimal) The numbering of fold if the model is sent by k-fold loop. Pipelined to be used for labeling.
# Output : fpr           = False positives that will be used to find threshold for the ROC-curve.
# Output : tpr           = True positives that will be used to find threshold for the ROC-curve.
# Output : threshold     = Thresholds created by the false positive-, and true positive rates. 
# Output : auc           = The area under curve value, meaning how many datapoints are correctly classified. 
# Output : label         = The name of the classification model used.
# Output : clean         = (Optimal) Cleaning parameters to be pipelined to draw_roc.
# Output : smalldata     = (Optimal) Bool variable to know which dataset is used. Pipelined for draw_roc.
# Output : vector_type   = (Optimal) The the vectorization form used for bag-of-words approach. Pipelined for labeling. 
def get_ROC(model, X_test, y_test, clean = None, smalldata = False, vector_type = None, i = None):
    # Get the probability of the predictions for spam only.
    if isinstance(model, LinearSVC):
        prob = model.decision_function(X_test)
    else:
        prob = model.predict_proba(X_test)
        prob = prob[:, 1]
    # Compute the AUROC, area under receiver operating characteristic
    auc = roc_auc_score(y_test, prob)
    y_test= '1' <= y_test
    # Get false positives and true positives to be used for plotting the ROC-curve
    fpr, tpr, threshold = roc_curve(y_test, prob)
    # Return model name to be used in drawing the ROC-curve.
    label = get_label(model, i)
    # If clean is present, the pipeline will include all parameters to label data.
    if clean is not None:
        return [fpr, tpr, threshold, auc, label, clean, smalldata, vector_type]
    else:
        return [fpr, tpr, threshold, auc, label]

# Plotting the ROC-curve with a 2D-list of ROC-data lists output by get_ROC().
# Either called immediately by train_model(), or called with a list which has multiple ROC-data lists built up by multiple train_model() calls.
# Param  : roc_list      = List of ROC-data. 
# Param  : X_test        = The test-data set which the model uses for validation. Needed for probability predictions.
# Output : roc_list      = Empties the lists for a more flexible call method.
def draw_ROC(roc_list, present):
    filename = None
    # For each list in roc_list, plot the respective ROC-curve using true positives and false positives rates.
    for roc_data in roc_list:
        plt.plot(roc_data[0], roc_data[1], marker='.', label=f" {roc_data[4]} with {len(roc_data[2])} points : AUC = {round(roc_data[3],2)}")
        # roc_data can only vary of being bigger than 5. If this is true, the pipelied is tasked to save figures.
        # Thus, it needs to gather correct labeling for filenames.
        if len(roc_data) is not 5:
            data_name = get_file_name(roc_data[5], roc_data[6], roc_data[7] ,"ROC")
            model_name = roc_data[4]
    plt.plot(plt.xlim(), plt.ylim(), linestyle="--")
    # Label our figure and corresponding x-axis and y-axis
    plt.title("ROC plot")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.legend()
    # If filename is present, the function saves the figure to /Screenshots
    if filename is not None:
        plt.savefig(f"Screenshots/{model_name}/{data_name}.png")
    # If present is true, show the plot. This is sent as a parameter in case we don't want the program to stop to show the figure.
    if present:
        plt.show()
    # Clear plot to make it ready for a new loop.
    plt.clf()
    # Empty list. 
    roc_list = []
    return roc_list

# Simple function which returns a model name. If sent in with a k-fold numbering, output label saying so.
# Param  : model         = Model to be named.
# Param  : i             = k-fold numbering.
# Output : label         = Label of the model recieve, in string format.
def get_label(model, i):
    if i is None:
        label = model.__class__.__name__
    else:
        label = f"{model.__class__.__name__} with fold {i}"
    return label

# Function that plots a confusion matrix.

def plot_cm(cm, model, clean, show, smalldata, vector_type, i = None):
    data_name = get_file_name(clean, smalldata, vector_type,"CM")
    df_cm = pd.DataFrame(cm, index = ["Real Ham", "Real Spam"], columns = ["Guessed Ham", "Guessed Spam"])
    sn.heatmap(df_cm, annot=True, fmt="d", cmap=sn.color_palette("rocket_r", as_cmap=True))
    plt.savefig(f"Screenshots/{get_label(model, i)}/{data_name}.png")
    if show:
        plt.show()
    plt.clf()

def get_file_name(clean, smalldata, vector, calltype):
    cleandata = ""
    dataframe = "B"
    caller = calltype
    vector_name = ""
    if isinstance(vector, TfidfVectorizer):
        vector_name = "_tfidf"
    if smalldata:
        dataframe = "S"
    if clean[0]:
        if clean[1] and clean[2]:
            cleandata = "CLS"
        elif clean[1]:
            cleandata = "CL"
        else:
            cleandata = "CS"
    else:
        cleandata = "U"
    return dataframe + "_" + caller + "_" + cleandata + vector_name
