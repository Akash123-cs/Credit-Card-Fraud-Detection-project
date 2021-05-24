#Importing basic packages/modules
import time
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.externals import joblib

#Load the dataset
data = pd.read_csv('creditcard.csv', delimiter=',')

#Data Pre-processing
#Data Extraction
positive = data.loc[data['Class']==1]
negative = data.loc[data['Class']==0]
negative = negative[400:850]

#Data preparation
cleaned_data = pd.concat([positive, negative])

#Splitting Data into Parameters & Target Classes
X = cleaned_data.iloc[:,0:29]
Y = cleaned_data.iloc[:,-1]

#Splitting the dataset into Train and Test data
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 25)


#1. Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier()
RFC.fit(xtrain, ytrain)
pred_rfc = RFC.predict(xtest)
acc_rfc = metrics.accuracy_score(ytest, pred_rfc)*100
#joblib.dump(RFC, 'Credit_RFC.pkl')

print('1. Using RandomForestClassifier Method')
print('Accuracy - {}'.format(acc_rfc))
print('Recall - {}'.format(metrics.recall_score(ytest, pred_rfc)))
print('Precision Score - {}'.format(metrics.precision_score(ytest, pred_rfc)))
print('Confusion matrix')
print(metrics.confusion_matrix(ytest, pred_rfc))
print('\n')
time.sleep(1)


#2. Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
GB = GradientBoostingClassifier()
GB.fit(xtrain, ytrain)
pred_gb = GB.predict(xtest)
acc_gb = metrics.accuracy_score(ytest, pred_gb)*100
#joblib.dump(GB, 'Credit_GB.pkl')

print('2. Using Gradient Boosting Method')
print('Accuracy - {}'.format(acc_gb))
print('Recall - {}'.format(metrics.recall_score(ytest, pred_gb)))
print('Precision Score - {}'.format(metrics.precision_score(ytest, pred_gb)))
print('Confusion matrix')
print(metrics.confusion_matrix(ytest, pred_gb))
print('\n')
time.sleep(1)


#3. Logistic Regression
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(xtrain, ytrain)
pred_lr = LR.predict(xtest)
acc_lr = metrics.accuracy_score(ytest, pred_lr)*100
#joblib.dump(LR, 'Credit_LR.pkl')

print('3. Using Logistic Regression Method')
print('Accuracy - {}'.format(acc_lr))
print('Recall - {}'.format(metrics.recall_score(ytest, pred_lr)))
print('Precision Score - {}'.format(metrics.precision_score(ytest, pred_lr)))
print('Confusion matrix')
print(metrics.confusion_matrix(ytest, pred_lr))
print('\n')
time.sleep(1)


#4. Support Vector Machines
from sklearn import svm
SVM = svm.LinearSVC(loss='hinge')
SVM.fit(xtrain, ytrain)
pred_svm = SVM.predict(xtest)
acc_svm = metrics.accuracy_score(ytest, pred_svm)*100
#joblib.dump(SVM, 'Credit_SVM.pkl')

print('4. Using SVM Method')
print('Accuracy - {}'.format(acc_svm))
print('Recall - {}'.format(metrics.recall_score(ytest, pred_svm)))
print('Precision Score - {}'.format(metrics.precision_score(ytest, pred_svm)))
print('Confusion matrix')
print(metrics.confusion_matrix(ytest, pred_svm))
print('\n')
time.sleep(1)


#5. K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=15)
KNN.fit(xtrain, ytrain)
pred_knn = KNN.predict(xtest)
acc_knn = metrics.accuracy_score(ytest, pred_knn)*100
#joblib.dump(KNN, 'Credit_KNN.pkl')

print('5. Using KNN Method')
print('Accuracy - {}'.format(acc_knn))
print('Recall - {}'.format(metrics.recall_score(ytest, pred_knn)))
print('Precision Score - {}'.format(metrics.precision_score(ytest, pred_knn)))
print('Confusion matrix')
print(metrics.confusion_matrix(ytest, pred_knn))
print('\n')
time.sleep(1)


#6. Decision Tree
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(max_depth=10, random_state=101, max_features=None, min_samples_leaf=10)
DT.fit(xtrain, ytrain)
pred_DT = DT.predict(xtest)
acc_DT = metrics.accuracy_score(ytest, pred_DT)*100
#joblib.dump(DT, 'Credit_DT.pkl')

print('6. Using Decision Tree Method')
print('Accuracy - {}'.format(acc_DT))
print('Recall - {}'.format(metrics.recall_score(ytest, pred_DT)))
print('Precision Score - {}'.format(metrics.precision_score(ytest, pred_DT)))
print('Confusion matrix')
print(metrics.confusion_matrix(ytest, pred_DT))
print('\n')
time.sleep(1)


#7. XGBoost
from xgboost import XGBClassifier
XGB = XGBClassifier()
XGB.fit(xtrain, ytrain)
pred_XGB = XGB.predict(xtest)
acc_XGB = metrics.accuracy_score(ytest, pred_XGB)*100
#joblib.dump(XGB, 'Credit_XGB.pkl')

print('7. Using XGBoost Method')
print('Accuracy - {}'.format(acc_XGB))
print('Recall - {}'.format(metrics.recall_score(ytest, pred_XGB)))
print('Precision Score - {}'.format(metrics.precision_score(ytest, pred_XGB)))
print('Confusion matrix')
print(metrics.confusion_matrix(ytest, pred_XGB))
print('\n')
time.sleep(1)


#8. MLP Classifier
from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(hidden_layer_sizes=(15, 4, 1))
MLP.fit(xtrain, ytrain)
pred_MLP = MLP.predict(xtest)
acc_MLP = metrics.accuracy_score(ytest, pred_MLP)*100
#joblib.dump(MLP, 'Credit_MLP.pkl')

print('8. Using MLP Method')
print('Accuracy - {}'.format(acc_MLP))
print('Recall - {}'.format(metrics.recall_score(ytest, pred_MLP)))
print('Precision Score - {}'.format(metrics.precision_score(ytest, pred_MLP)))
print('Confusion matrix')
print(metrics.confusion_matrix(ytest, pred_MLP))
print('\n')
time.sleep(1)


#Plotting the Accuracies achieved
import matplotlib.pyplot as plt

#Defining labels and data
height = [acc_rfc, acc_gb, acc_lr, acc_svm , acc_knn, acc_DT, acc_XGB, acc_MLP]
bars = ('RF', 'GB', 'LR', 'SVM', 'KNN', 'DT', 'XGB', 'MLP')
y_pos = np.arange(len(bars))
 
# Create bars and choose color
plt.bar(y_pos, height, color = (1.00, 0.65, 0.00, 1.0))
 
# Add Title and Axis names
plt.title('Comparision Study')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
 
# Limits for the Y axis
plt.ylim(0,100) 
# Create names
plt.xticks(y_pos, bars) 
#Save the graphic
plt.savefig('Comparision_Study')
# Show graphic
plt.show()



#For real time predictions
#Input to this function can be a 1-D Array of 29 parameters
#ex = np.array(cleaned_data.iloc[0]) or np.array(cleaned_data.iloc[500])
#ex = ex[0:29]
print('A Real Time Prediction..\n')
def new_predict(i):
    print(i)
    a = np.array([i])
    pred = SVM.predict(a)
    result = round(pred[0])
    if result==0:
        print('Fraud Transaction - Negative')
    else:
        print('Fraud Transaction - Positive')
      
ex = np.array(cleaned_data.iloc[0])
ex = ex[0:29]
new_predict(ex)