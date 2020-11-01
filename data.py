# Data retrieval
import pandas as pd

# Read the csv file with all the emails
df = pd.read_csv('data/emails.csv')
exclusions = pd.read_csv('data/exclusions.csv')['exclusions'].to_list()

# Number of spam and ham emails in the dataset
num_spam = df['label'].value_counts()[0]
num_ham = df['label'].value_counts()[1]

# Dictionary with most often occuring words for spam and ham emails
words_spam = {} 
words_ham = {}


# Returns the entire dataframe
def get_data():
    return df


# Returns a tuple with the email and its label at the index
def get_row(index):
    return df.at[index, 'email'], df.at[index, 'label']


# Just for fun, returns num of words in an email at given index
def num_words(index):
    email = get_row(index)[0]
    return len(email.split())


# Data cleaning
def clean_data(max_length = 3):
    temp = ""
    
    for index, row in df.iterrows():
        for word in str(row['email']).lower().split():            
            if len(word) > max_length and (word not in exclusions):
                temp = temp + word.lower() + " "
        df.at[index, 'email'] = temp
        temp = ""


# Clears and then populates the dictionaries 
def count_words():
    clean_data()
    
    words_ham.clear()
    words_spam.clear()
    
    # The function that does the counting
    def word_counter(email, words_dict):
        for word in str(email).split():
            if word in words_dict:
                words_dict[word] = words_dict[word] + 1
            else:
                words_dict[word] = 1

    # Looping through each item in the emails dataset
    for index, row in df.iterrows():
        # Check the label first
        if row['label'] == 1: 
            word_counter(row['email'], words_spam )
        elif row['label'] == 0:
            word_counter(row['email'], words_ham )
        else:
            print("Wrong label used - " + str(row['label']) + " at index " + str(index) + ".  Ignoring it." )

            

