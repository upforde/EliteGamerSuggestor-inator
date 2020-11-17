# Importing the algorithm implementations
import naive as alg
import data

# Plotting tools
import matplotlib.pyplot as plt




def TPR(TP, FN):
    return TP / (TP + FN)

def FPR(FP, TN):
    return FP / (FP + TN)





plt.figure()

plt.plot([0,0], [1,1])

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")


plt.show()