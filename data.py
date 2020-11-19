# Data retrieval
import pandas as pd
import os

# Natural language processing tookit
import nltk.stem 
from nltk.corpus import stopwords

import string
import re

# If any of the nltk libraries are not already downloaded, uncomment this
#nltk.download('wordnet')
#nltk.download('stopwords')

# Read the csv file with all the Spamassassin emails
# All email are read into a pandas dataframe
df = pd.read_csv('data/emails.csv')

# Enron database 

# Returns the Enron db
# lower cased, punctuation and newlines removed, urls and numbers replaces to words "url" and "number"
def get_big_df():
    
    def is_number(n):
        try:
            float(n)   
        except ValueError:
            return False
        return True

    def pre_clean(email):
        # Remove the subject filed
        email = email.split('\n', 1)[1]
        
        # Lowercase the email, and remove all newlines
        email = email.lower().replace('\n', ' ')
        
        # The regex is from https://daringfireball.net/2010/07/improved_regex_for_matching_urls 
        # Swaps links with the word "url"
        url = re.compile(r"""(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))""")    
        email = url.sub(r'url', email)
    
        temp = ""

        # Replaces all numbers with the word "number"
        for word in email.split():
            if is_number(word):
                word = "number"

            temp = temp + word + " "
        
        return temp.translate(temp.maketrans("","", string.punctuation))

    # Create a new dataframe that will be returned
    big_df = pd.DataFrame(columns=('email', 'label'))
    
    # Adding all ham emails
    for entry in os.scandir(r'data/Enron/ham'):
        big_df = big_df.append( pd.DataFrame( [[pre_clean(open(entry, errors=('replace')).read()), 0]] , columns = ['email', 'label']), ignore_index=True )
    
    # Adding all spam emails
    for entry in os.scandir(r'data/Enron/spam'):
        big_df = big_df.append( pd.DataFrame( [[pre_clean(open(entry, errors=('replace')).read()), 1]] , columns = ['email', 'label']), ignore_index=True )
        
    
    return big_df

big_df = get_big_df()

# The stop words from nltk library has the most common words used in a language that bear no meaning (articles, prepositions, etc.)
stop_words = set(stopwords.words('english'))


# Number of spam and ham emails in the dataset for easy retrieval
def num_spam(is_df = True):
    if is_df:
        df['label'].value_counts()[0]
    else:
        big_df['label'].value_counts()[0]

def num_ham(is_df = True):
    if is_df:        
        df['label'].value_counts()[1]
    else:
        big_df['label'].value_counts()[1]
        
def num_total(is_df = True): 
    return num_ham(is_df) + num_spam(is_df)

# Dictionary with most often occuring words for spam and ham emails
# Not proud of this, but trying to keep the work of changing files using this one to the minimum

# Consist of value-key pair word: total_used 
df_spam = {} 
df_ham = {}

big_df_spam = {}
big_df_ham = {}

# Returns the entire dataframe for easy access
def get_data():
    return df

# Returns a tuple with the email and its label at the index
def get_row(index):
    return df.at[index, 'email'], df.at[index, 'label']


# Returns num of words in an email at given index
def num_words(index):
    email = get_row(index)[0]
    return len(email.split())

# Data cleaning is a preprocessing step that:
    # 1. Converts everything to lower case; 
    # 2. Removes words shorter than max_length; 
    # 3. Removes words given in the exclusion list;
    # 4. Lemmatizes the words;
    # 5. Stemming 
    
def clean_data(max_length = 1, lemmatize = True, stem = True, df = True):
    temp = ""
    lemmatizer = nltk.stem.WordNetLemmatizer()
    stemmer = nltk.stem.PorterStemmer()

    if df:
        database = df
    else:
        database = big_df

    # Loop through emails
    for index, row in database.iterrows():

        # Lower case the entire email
        email_lc = str(row['email']).lower()
        
        # Split the email and loop through each word
        for word in email_lc.split():
            # If the word passes the length check and is not in the exclustions list it is added back into the email
            if len(word) > max_length and (word not in stop_words):
                temp = temp + word + " "

        # Lemmatize
        if lemmatize:
            lemma_temp = ""
            for word in temp.split():
                lemma_temp = lemma_temp + lemmatizer.lemmatize(word) + " "
            
            temp = lemma_temp

        # Stemming
        if stem:
            stem_temp = ""
            for word in temp.split():
                stem_temp = stem_temp + stemmer.stem(word) + " "
            
            temp = stem_temp
        
        # Rewrite the original email 
        database.at[index, 'email'] = temp
        temp = ""

# Clears and then populates the dictionaries 
def count_words(data_cleaning = True, lemmatize = True, stem = True, df = True):
    
    if data_cleaning:
        clean_data(data_cleaning, lemmatize, stem, df)

    # The function that does the counting
    def word_counter(email, words_dict):
         
        for word in str(email).split():
            if word in words_dict:
                words_dict[word] = words_dict[word] + 1
            else:
                words_dict[word] = 1

    def iterate(dic_spam, dic_ham, database):
        # Looping through each item in the emails dataset
        for index, row in database.iterrows():
            # Check the label first
            if row['label'] == 1: 
                word_counter(row['email'], dic_spam )
            elif row['label'] == 0:
                word_counter(row['email'], dic_ham )
            else:
                print("Wrong label used - " + str(row['label']) + " at index " + str(index) + ".  Ignoring it." )
                    
    if df:
        df_ham.clear()
        df_spam.clear()
        iterate(df_spam, df_ham, df)
    else:
        big_df_ham.clear()
        big_df_spam.clear()
        iterate(big_df_spam, big_df_ham, big_df)

            
# Orders the words and return an ordered list 
# The elements are lists with the word at index 0 and num of occurences at index 1
# The list is ordered, so the first value is the most occured word
def order_words(df = True):
    if df:
        database = df
    else:
        database = big_df
    
    if (database == df):
        return sorted(df_ham.items(), key=lambda x: x[1], reverse = True) , sorted(df_spam.items(), key=lambda x: x[1], reverse = True)
    else:
        return sorted(big_df_ham.items(), key=lambda x: x[1], reverse = True) , sorted(big_df_spam.items(), key=lambda x: x[1], reverse = True)
 