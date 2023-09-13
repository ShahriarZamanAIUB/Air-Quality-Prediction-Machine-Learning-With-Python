import matplotlib.pyplot as plt 
from sklearn import __version__
from sklearn import metrics

from sklearn import datasets
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
 
 

import pandas as pd
df = pd.read_csv('airdata.csv')

print("\n\nShape of DataFrame:", df.shape)

# The notna() method detects the empty cells
# for non-empty cells it return True and vice-versa
df = df[df['Country'].notna()]
print(df)

"""
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df['AQI'] = scaler.fit_transform(df['AQI'].values.reshape(-1,1)).round(3)

scaler = MinMaxScaler()
df['Ozone_Value'] = scaler.fit_transform(df['Ozone_Value'].values.reshape(-1,1)).round(3)

scaler = MinMaxScaler()
df['NO2_Value'] = scaler.fit_transform(df['NO2_Value'].values.reshape(-1,1)).round(3)

scaler = MinMaxScaler()
df['PM_2.5_Value'] = scaler.fit_transform(df['PM_2.5_Value'].values.reshape(-1,1)).round(3)

"""
"""
correlation_matrix = df.corr().round(2)
# changing the figure size
plt.figure(figsize = (10, 7))
# "annot = True" to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()


# Countplot will show barcharts based on categories
# of selected feature
import seaborn as sns
b=sns.countplot(x='AQI_Category', data=df)
b.set_xlabel("Air Quality Index Status",fontsize=15)
b.set_ylabel("No. of Records",fontsize=15)
plt.title("All gradings of AQI count",fontsize=20)
plt.show()

# Countplot will show barcharts based on categories
# of selected feature
import seaborn as sns
b=sns.countplot(x='Ozone_Category', data=df)
b.set_xlabel("Ozone Count Status",fontsize=15)
b.set_ylabel("No. of Records",fontsize=15)
plt.title("All gradings of Ozone count",fontsize=20)
plt.show()




# Countplot will show barcharts based on categories
# of selected feature
import seaborn as sns
b=sns.countplot(x='CO_Category', data=df)
b.set_xlabel("CO Status of Countries",fontsize=18)
b.set_ylabel("No. of Records",fontsize=18)
plt.title("Gradings of Carbon Monoxide for all Countries",fontsize=22)
plt.show()

# Countplot will show barcharts based on categories
# of selected feature
import seaborn as sns
sns.countplot(x='NO2_Category', data=df)
plt.title("All Categories of NO2 count")
plt.show()

import seaborn as sns
sns.countplot(x='PM_2.5_Category', data=df)
plt.title("All gradings of PM 2.5 count")
plt.show()

"""


# We are deleting the following redundant features from df

del df['Country']  #Deleting Country names
del df['City'] # Deleting City names
del df['AQI']
del df['CO_Category'] # Deleting Carbon Monoxide Status
del df['Ozone_Category'] # Deleting Ozone Status
del df['NO2_Category']  # Deleting Nitrous Oxide Status
del df['PM_2.5_Category'] # Deleting Particulate Status


# let's visualise the number of samples for each class with count plot

"""


# Let us make a scatterplot to see the linear relationship 
# between Air Quality Index and Ozone Value
plt.figure(figsize=(5,5))

 
b=sns.scatterplot(x=df['Ozone_Value'], y=df['AQI'])
b.set_xlabel("Value of Ozone in Air",fontsize=18)
b.set_ylabel("Air Quality Index",fontsize=18)
 
plt.title("Air Quality Index vs Ozone Value",fontsize=22)
 

# Let us make a scatterplot to see the linear relationship 
# between Air Quality Index and Carbon Monoxide Value
plt.figure(figsize=(5,5))

b=sns.scatterplot(x=df['CO_Value'], y=df['AQI'])
b.set_xlabel("Value of Carbon Monoxide in Air",fontsize=18)
b.set_ylabel("Air Quality Index",fontsize=18)
 
plt.title("Air Quality Index vs CO Value",fontsize=22)

# Let us make a scatterplot to see the linear relationship 
# between Air Quality Index and Nitrous oxide Value
plt.figure(figsize=(5,5))

b=sns.scatterplot(x=df['NO2_Value'], y=df['AQI'])
b.set_xlabel("Value of Nitrous oxide in Air",fontsize=18)
b.set_ylabel("Air Quality Index",fontsize=18)
 
plt.title("Air Quality Index vs NO2 Value",fontsize=22)


# Let us make a scatterplot to see the linear relationship 
# between Air Quality Index and 2.5 micron Particulates Value
plt.figure(figsize=(5,5))

b=sns.scatterplot(x=df['PM_2.5_Value'], y=df['AQI'])
b.set_xlabel("Value of Particulates in Air",fontsize=18)
b.set_ylabel("Air Quality Index",fontsize=18)
 
plt.title("Air Quality Index vs Particulates Value",fontsize=22)


plt.show()

"""
"""
#print(df["Ozone_Category"].unique())
for x in df.index:
 if df.loc[x, "Ozone_Category"] == 'Hazardous':
  df.loc[x, "Ozone_Category"] = 5
 elif df.loc[x, "Ozone_Category"] == 'Very Unhealthy':
  df.loc[x, "Ozone_Category"] = 4  
 elif df.loc[x, "Ozone_Category"] == 'Unhealthy':
  df.loc[x, "Ozone_Category"] = 3
 elif df.loc[x, "Ozone_Category"] == 'Unhealthy for Sensitive Groups':
  df.loc[x, "Ozone_Category"] = 2
 elif df.loc[x, "Ozone_Category"] == 'Moderate':
  df.loc[x, "Ozone_Category"] = 1
 elif df.loc[x, "Ozone_Category"] == 'Good':
  df.loc[x, "Ozone_Category"] = 0
 
"""

df.dropna(inplace=True) 
#print(df)
df.to_csv('clean_dataset.csv')

y = df['AQI_Category'].to_numpy() # Air Quality Index is our Target Vector 
 

del df['AQI_Category']
X = df.to_numpy()  

 
# importing train_test_split method from model_selection module
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
random_state = 1)
 # test_size = 0.2 means 80% of our rows will be used for training and 20% of our rows will be used for testing
 # random_state = 1 guarantees hat the split will always be the same (i.e. reproducible results)
print("Data has been split!\n")
print("X_train shape: ", X_train.shape) # Printing dimensions of training and testing data
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)


 

############################
from sklearn import svm 
model_svm = svm.SVC(class_weight='balanced') #select the algorithm
model_svm.fit(X_train, y_train) #train the model with the training dataset
y_prediction_svm = model_svm.predict(X_test) # pass the testing data to the trained model
# checking the accuracy of the algorithm. 
# by comparing predicted output by the model and the actual output
score_svm = metrics.accuracy_score(y_prediction_svm, y_test).round(4)



print("----------------------------------")
print('The accuracy of the Support Vector Machine is: {}'.format(score_svm))
print("----------------------------------")

#recall_svm = metrics.recall_score(y_prediction_svm, y_test).round(4)
print("---------------------------------")
#print('The recall of the SVM is: {}'.format(recall_svm))
#print("---------------------------------")

#precision_svm = metrics.precision_score(y_prediction_svm, y_test).round(4)
#print("---------------------------------")
#print('The precision of the SVM is: {}'.format(precision_svm ))
# save the accuracy score
score = set()
score.add(('SVM', score_svm))

print(confusion_matrix(y_test, y_prediction_svm).ravel())
#precision_svm=float((tp)/(tp+fp))
#recall_svm=float((tp)/(tp+fn))

#print(f"SVM, tp: {tp} fp: {fp} fn: {fn} tn: {tn} precision: {precision_svm} recall : {recall_svm}")




# importing the necessary package to use the classification algorithm
from sklearn.neighbors import KNeighborsClassifier # for K nearest neighbours
#from sklearn.linear_model import LogisticRegression # for Logistic Regression algorithm
model_knn = KNeighborsClassifier(n_neighbors=48 ) # n=number of samples=2291 rows, 
# n^0.5 neighbours chosen for putting the new data into a class
model_knn.fit(X_train, y_train) #train the model with the training dataset
y_prediction_knn = model_knn.predict(X_test) #pass the testing data to the trained model
# checking the accuracy of the algorithm. 
# by comparing predicted output by the model and the actual output
score_knn = metrics.accuracy_score(y_prediction_knn, y_test).round(4)
print("----------------------------------")
print('The accuracy of the K-Nearest Neighbour is: {}'.format(score_knn))
print("----------------------------------")
# save the accuracy score
score.add(('KNN', score_knn))

print(confusion_matrix(y_test, y_prediction_knn).ravel())
#precision_knn=float((tp)/(tp+fp))
#recall_knn=float((tp)/(tp+fn))

#print(f"KNN, tp: {tp} fp: {fp} fn: {fn} tn: {tn} precision: {precision_knn} recall : {recall_knn}")


# importing the necessary package to use the classification algorithm
from sklearn.linear_model import LogisticRegression # for Logistic Regression algorithm
model_lr = LogisticRegression(max_iter=30000, class_weight='balanced')
model_lr.fit(X_train, y_train) #train the model with the training dataset
y_prediction_lr = model_lr.predict(X_test) #pass the testing data to the trained model
# checking the accuracy of the algorithm. 
# by comparing predicted output by the model and the actual output
score_lr = metrics.accuracy_score(y_prediction_lr, y_test).round(4)
print("---------------------------------")
print('The accuracy of the Logistic Regression is: {}'.format(score_lr))
print("---------------------------------")
# save the accuracy score
score.add(('LR', score_lr))


#print(confusion_matrix(y_test, y_prediction_lr).ravel())
#precision_LR=float((tp)/(tp+fp))
#recall_LR=float((tp)/(tp+fn))

#print(f"LR, tp: {tp} fp: {fp} fn: {fn} tn: {tn} precision: {precision_LR} recall : {recall_LR}")


# importing the necessary package to use the classification algorithm
from sklearn.naive_bayes import GaussianNB
model_nb = GaussianNB( )
model_nb.fit(X_train, y_train) #train the model with the training dataset
y_prediction_nb = model_nb.predict(X_test) #pass the testing data to the trained model
# checking the accuracy of the algorithm. 
# by comparing predicted output by the model and the actual output
score_nb = metrics.accuracy_score(y_prediction_nb, y_test).round(4)
print("---------------------------------")
print('The accuracy of the Naive Bayes Classifier is: {}'.format(score_nb))
print("---------------------------------")
# save the accuracy score
score.add(('NB', score_nb))

"""
tn, fp, fn, tp = confusion_matrix(y_test, y_prediction_nb).ravel()
precision_NB=float((tp)/(tp+fp))
recall_NB=float((tp)/(tp+fn))

print(f"NB, tp: {tp} fp: {fp} fn: {fn} tn: {tn} precision: {precision_NB} recall : {recall_NB}")
"""

# importing the necessary package to use the classification algorithm
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
model_dt = DecisionTreeClassifier( )
model_dt.fit(X_train, y_train) #train the model with the training dataset
y_prediction_dt = model_dt.predict(X_test) #pass the testing data to the trained model
# checking the accuracy of the algorithm. 
# by comparing predicted output by the model and the actual output
score_dt = metrics.accuracy_score(y_prediction_dt, y_test).round(4)
print("---------------------------------")
print('The accuracy of the Decision Tree Classifier is: {}'.format(score_dt))
print("---------------------------------")
# save the accuracy score
score.add(('DT', score_dt))

"""
tn, fp, fn, tp = confusion_matrix(y_test, y_prediction_dt).ravel()
precision_DT=float((tp)/(tp+fp))
recall_DT=float((tp)/(tp+fn))

print(f"DT, tp: {tp} fp: {fp} fn: {fn} tn: {tn} precision: {precision_DT} recall : {recall_DT}")
"""


print("The accuracy scores of different Models:")
print("----------------------------------------")
count=0
sum=0
for s in score:
 print(s)


import numpy as np
#Making 2 numpy arrays for our model names and their classification accuracies
model_names = np.array(['SVM', 'KNN', 'Logistic Regression', 'Naive Bayes', 'Decision Tree'])
model_accuracies = np.array([score_svm, score_knn, score_lr, score_nb, score_dt])

model_names_axis = np.arange(len(model_names))#List of index, will be used in labelling
 
# Making bar plot with unique color for each bar
plt.bar(model_names,model_accuracies, color=['purple', 'red', 'green', 'blue', 'orange'])

# Setting labels below each bar 
plt.xticks(model_names_axis, model_names, fontsize=15)
plt.xlabel("Models",fontsize=20) # X axis label
plt.ylabel("Accuracy",fontsize=20) # Y axis label
plt.title("Classification Accuracy of Alll the Classifiers used",fontsize=22) # Title
 
plt.show()

#SVM confusion matrix


 
 
cm_svm = confusion_matrix(y_test, y_prediction_svm)
sns.heatmap(cm_svm, annot=True, cmap='Greens', xticklabels=['Good','Hazardous','Moderate','Unhealthy','UFSG','Very Unhealthy'], yticklabels=['Good','Hazardous','Moderate','Unhealthy','UFSG','Very Unhealthy'])

plt.xlabel('Predicted labels(SVM)',fontsize=14)
plt.xticks(rotation=0,   fontsize=12)
plt.yticks(rotation=0,   fontsize=12)
plt.ylabel('True labels(SVM)',fontsize=14)
plt.title('Confusion Matrix(SVM)',fontsize=18)
 
 
plt.show()




#KNN confusion matrix
cm_knn = confusion_matrix(y_test, y_prediction_knn)
 

 
sns.heatmap(cm_knn, annot=True, cmap='Purples', xticklabels=['Good','Hazardous','Moderate','Unhealthy','UFSG','Very Unhealthy'], yticklabels=['Good','Hazardous','Moderate','Unhealthy','UFSG','Very Unhealthy'])

plt.xlabel('Predicted labels(KNN)',fontsize=14)
plt.xticks(rotation=0,   fontsize=12)
plt.yticks(rotation=0,   fontsize=12)
plt.ylabel('True labels(KNN)',fontsize=14)
plt.title('Confusion Matrix(KNN)',fontsize=18)


 
plt.show()


#DT confusion matrix
cm_dt = confusion_matrix(y_test, y_prediction_dt)
 

 
sns.heatmap(cm_dt, annot=True, cmap='Greys', xticklabels=['Good','Hazardous','Moderate','Unhealthy','UFSG','Very Unhealthy'], yticklabels=['Good','Hazardous','Moderate','Unhealthy','UFSG','Very Unhealthy'])

plt.xlabel('Predicted labels(Decision Tree)',fontsize=14)
plt.xticks(rotation=0,   fontsize=12)
plt.yticks(rotation=0,   fontsize=12)
plt.ylabel('True labels(Decision Tree)',fontsize=14)
plt.title('Confusion Matrix(Decision Tree)',fontsize=18)


 
plt.show()

#Naive Bayes confusion matrix

cm_nb = confusion_matrix(y_test, y_prediction_nb)
 

 
sns.heatmap(cm_nb, annot=True, cmap='YlOrBr', xticklabels=['Good','Hazardous','Moderate','Unhealthy','UFSG','Very Unhealthy'], yticklabels=['Good','Hazardous','Moderate','Unhealthy','UFSG','Very Unhealthy'])

plt.xlabel('Predicted labels(Naive Bayes)',fontsize=14)
plt.xticks(rotation=0,   fontsize=12)
plt.yticks(rotation=0,   fontsize=12)
plt.ylabel('True labels(Naive Bayes)',fontsize=14)
plt.title('Confusion Matrix(Naive Bayes)',fontsize=18)


 
plt.show()


#Logistic Regression confusion matrix

cm_lr = confusion_matrix(y_test, y_prediction_lr)
 

 
sns.heatmap(cm_lr, annot=True, cmap='Reds', xticklabels=['Good','Hazardous','Moderate','Unhealthy','UFSG','Very Unhealthy'], yticklabels=['Good','Hazardous','Moderate','Unhealthy','UFSG','Very Unhealthy'])

plt.xlabel('Predicted labels(Logistic Regression)',fontsize=14)
plt.xticks(rotation=0,   fontsize=12)
plt.yticks(rotation=0,   fontsize=12)
plt.ylabel('True labels(Logistic Regression)',fontsize=14)


 
plt.show()

#y_test_df = pd.DataFrame(y_test, columns = ['Test'])
#y_test_df.to_csv('datasetx_processed2.csv')



 

 

 


