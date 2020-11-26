### :e-mail: Spam-classification using Naive Bayes, Support Vector Machines, or Perceptron :e-mail:

A Machine Learning project for TDT4173 - Machine Learning done by 
- Piri Babayev
- Danilas Miscenko
- Aleksander Simmersholm.


### Directory :file_folder:

- Screenshots :open_file_folder:
  - BernoulliNB 
    - ...
  - LinearSVC
    - ...
  - MultinomialNB
    - ...
  - Perceptron
    - ...
- data :open_file_folder:
  - Enron
    - ...
  - Spamassassin rest
    - ...
  - emails.csv
  - exclusions.csv
- perceptron.py
- data.py
- methods.py
- visuals.py


### Directory legend :arrow_right_hook:

- **Screenshots** is the directory of all images produced by the code files which are discussed and used in the project report.
  - Each classifer has their own folder which includes the confusion matrices and ROC-curve plots with all possible permutations.
- **data** is the directory containing the dataset we run our classifiers on.
  - The **enron** dataset, which contains over 33000 e-mails.
  - **emails.csv**, the [kaggle dataset](https://www.kaggle.com/ozlerhakan/spam-or-not-spam-dataset) which the project originally started with.
  - **exclusions.csv**, words which are removed during data cleaning. 
- **perceptron.py**, The python file which includes the manually coded perceptron and all its necessary functions. 
- **data.py**, The python file which includes all data-cleaning functions. 
- **methods.py**, The python file which includes functions for training the Naive Bayes,- and Support Vector Machine classifiers through the scikit libraries.
- **visuals.py**, The python file which includes plotting functions for the methods.py file.


### How to run the code for Naive Bayes and SVM. :capital_abcd:

The main module for running Multinomial Naive Bayes, Bernoulli Naive Bayes, and Support Vector Machines is the **methods.py** file.
Each function is commented with an explaination for what it does, and what parameters it takes.
However, the only method one needs to use is the *main method*, which is the **train_model()** method.

### How to use the train_model() in methods.py

The train_model() trains a given model accordingly with numerous parameters to allow permutations. 

A except from the methods.py comment:
```
Param  : model         = Classification model to be trained and used for predictions and plotting.
Param  : smalldata     = If true, the small dataframe will be used. Otherwise big dataframe will be used.
Param  : clean         = List of three bool values to determine if [0]cleaning, [1]lemmatization, and [2]stemming is to be done.
Param  : vector_type   = Vectorization type to be used when performing bag-of-words for training the models.
Param  : n_folds       = Integer that determines number of k-folds. If less than 2, performs, 1/3 test/training split.
Param  : report        = If true, the function should output everything necessary for the project. Plots and saves everything to /Screenshots.
Param  : collect_roc   = If true, the function should collect and return a second output in the form of a list to be used for with the draw_roc() function. 
Param  : show          = If true, shows the plotted visuals as they're called. This however stops function execution till the window is exited. 
Output : model         = Returns the now trained classification model that got sent in from the first parameter.
Output : roc_data_list = (Optimal) ROC-curve data to be further pipelined into the ROC-curve plotting function.

def train_model(model, smalldata, clean, vector_type, n_folds, report, collect_roc, show)
```
There exists a few uncommented example lines close to the bottom to get a bearing on how to use it.

#### To get the same results as given in the project report for NB and SVM, do the following:


1. Comment out line 251 to line 266 by wrapping the lines inbetween like so:
```
"""
*...some code here...*

"""
```
2. Delete comment notation from line 269, line 342, line 348, and line 416. So the permutations are allowed to run.
3. Delete the image folder as these lines of codes will generate them all over again
4. Run the file! This will take about 45 minutes...
5. Hesitate. 

#### How to run the code for the Perceptron :eyes: