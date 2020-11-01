# Importing the data module
import data
import matplotlib.pyplot as plt


data.count_words()

words_ham = sorted(data.words_ham.items(), key=lambda x: x[1], reverse = True)
words_spam = sorted(data.words_spam.items(), key=lambda x: x[1], reverse = True)


print("Top Ham Words: ")

for i in range(1,100):
    print(str(i) + ".   " + words_ham[i][0] + " - " + str(words_ham[i][1]) )
    
    
print("\nTop Spam Words: \n")

for i in range(1,100):
    print(str(i) + ".   " + words_spam[i][0] + " - " + str(words_spam[i][1]))