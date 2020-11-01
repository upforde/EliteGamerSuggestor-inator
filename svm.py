# Importing the data module
import data


data.count_words()

words_ham = sorted(data.words_ham.items(), key=lambda x: x[1], reverse = True)
words_spam = sorted(data.words_spam.items(), key=lambda x: x[1], reverse = True)


print("Top 10 Ham Words: ")

for i in range(1,110):
    print(str(i) + ".   " + words_ham[i][0])
    
    
print("Top 10 Spam Words: ")

for i in range(1,110):
    print(str(i) + ".   " + words_spam[i][0])