# Data retrieval
import pandas as pd

# Read the csv file with all the emails
df = pd.read_csv('emails.csv')

# Number of spam and ham emails in the dataset
num_spam = df['label'].value_counts()[0]
num_ham = df['label'].value_counts()[1]

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



# <====================== Maybe useful

# Can also iterate like so, but bleh
# Return a typle with the email and its label
"""
for index, row in df.iterrows():
    row['email'], row['label']
"""