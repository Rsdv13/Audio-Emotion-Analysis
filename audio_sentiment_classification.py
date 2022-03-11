import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import pickle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


mslen = 22050

data = []

max_fs = 0
labels = []

emotions = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']

f2 = open('feature.pkl','rb')
feature_all = pickle.load(f2)
f3 = open('label.pkl','rb')
labels = pickle.load(f3)
from copy import deepcopy
y = deepcopy(labels)
for i in range(len(y)):
    y[i] = int(y[i])


n_labels = len(y)
n_unique_labels = len(np.unique(y))
one_hot_encode = np.zeros((n_labels,n_unique_labels))
f = np.arange(n_labels)
for i in range(len(f)):
    one_hot_encode[f[i],y[i]-1]=1


X_train,X_test,y_train,y_test = train_test_split(feature_all,one_hot_encode,test_size = 0.3,random_state=20)

########################### MODEL 1 ###########################



X_train, X_test, y_train, y_test = train_test_split(feature_all, y, test_size=0.3, random_state=2)



rbf_svm = SVC(kernel='poly', degree=12, C=5000).fit(X_train, y_train)

# In[172]:


svm_predictions = rbf_svm.predict(X_test)

# In[189]:


accuracy = rbf_svm.score(X_test, y_test)
print("Accuracy using SVM Model:",accuracy)

# In[190]:


cm = confusion_matrix(y_test, svm_predictions)

# In[191]:


########################### MODEL 2 ###########################


# In[192]:


X_train, X_test, y_train, y_test = train_test_split(feature_all, y, test_size=0.3, random_state=2)




classifier = RandomForestClassifier(n_estimators=70, criterion='entropy', random_state=42)
classifier.fit(X_train, y_train)




y_pred = classifier.predict(X_test)




accuracy_score(y_test, y_pred)
print("Accuracy using RandomForest Model:",accuracy_score(y_test, y_pred))





########################### MODEL 3 ###########################



X_train, X_test, y_train, y_test = train_test_split(feature_all, y, test_size=0.3, random_state=2)



knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)
print("Accuracy using KNN Model:", accuracy)

# creating a confusion matrix
knn_predictions = knn.predict(X_test)
cm = confusion_matrix(y_test, knn_predictions)

