# Data retrieval
import pandas as pd
import nltk.stem 


# TODO - test if this is required
# nltk.download('wordnet')

# Read the csv file with all the emails
# All email are read into a pandas dataframe
df = pd.read_csv('data/emails.csv')

# Exclusions are a list of words that are removed from the original data - these consist of stop words, articles, prepositions, etc.
exclusions = pd.read_csv('data/exclusions.csv')
# Convert into a list for faster processing down the line
exclusions = exclusions['exclusions'].to_list()

# Number of spam and ham emails in the dataset for easy retrieval
num_spam = df['label'].value_counts()[0]
num_ham = df['label'].value_counts()[1]

# Dictionary with most often occuring words for spam and ham emails
# Consist of value-key pair word: total_used 
words_spam = {} 
words_ham = {}


# Returns the entire dataframe for easy access
def get_data():
    return df


# Returns a tuple with the email and its label at the index
def get_row(index):
    return df.at[index, 'email'], df.at[index, 'label']


# Just for fun, returns num of words in an email at given index
def num_words(index):
    email = get_row(index)[0]
    return len(email.split())


# Data cleaning is a preprocessing step that:
    # 1. Converts everything to lower case; 
    # 2. Removes words shorter than max_length; 
    # 3. Removes words given in the exclusion list;
    # 4. Lemmatizes the words.
    
def clean_data(max_length = 3, lemmatize = False, stem = True):
    temp = ""
    lemmatizer = nltk.stem.WordNetLemmatizer()
    stemmer = nltk.stem.PorterStemmer()
    
    
    for index, row in df.iterrows():

        # Lower case the entire email
        email_lc = str(row['email']).lower()
        
        for word in email_lc.split():
            # If the word passes the length check and is not in the exclustions list it is added back into the email
            if len(word) > max_length and (word not in exclusions):
                temp = temp + word + " "

        # Lemmatize the email

        if lemmatize == True:
            temp = lemmatizer.lemmatize(temp)

        # Extra step - stemming
        if stem == True:
            stem_temp = ""
            for word in temp.split():
                stem_temp = stem_temp + stemmer.stem(word) + " "
            
            temp = stem_temp
        

        # Rewrite the original email 
        df.at[index, 'email'] = temp
        temp = ""


# Clears and then populates the dictionaries 
def count_words(data_cleaning = True):
    
    if data_cleaning == True:
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

            

