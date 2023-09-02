#!/usr/bin/env python
# coding: utf-8

# # Nitesh_Assessment 

# In[326]:


#Importing all Important liabries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().system('pip install matplotlib -U')


# In[299]:


import warnings
warnings.filterwarnings('ignore')

get_ipython().system('pip install Sklearn -U')


# In[300]:


#importing CSV file
data = pd.read_csv("C:\\Users\\USER\\Downloads\\BankChurners.csv.zip")


# In[301]:


#viewing data 
data.head()


# In[302]:


data.shape


# In[303]:


data.columns


# In[304]:


coloums = ["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1", "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"]


# In[305]:


data.drop (columns = coloums, inplace = True)


# In[306]:


data.head()


# In[307]:


data.drop("CLIENTNUM",axis = 1,inplace= True)


# In[308]:


data


# In[309]:


data.describe().T


# In[310]:


data.isnull().sum()


# In[311]:


X = data[["Attrition_Flag", "Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"]]


# In[312]:


X["Gender"].unique()


# In[313]:


data.info()


# In[314]:


data.head()


# In[315]:


data.nunique()


# In[316]:


data.info()


# In[331]:


data.duplicated()


# In[318]:


from sklearn.preprocessing import LabelEncoder 


# In[319]:


le = LabelEncoder()


# In[320]:


data["Attrition_Flag"] = le.fit_transform (data["Attrition_Flag"])


# In[321]:


data.head()


# In[322]:


data["Gender"] = le.fit_transform(data["Gender"])
data["Education_Level"] = le.fit_transform(data["Education_Level"])
data["Marital_Status"] = le.fit_transform(data["Marital_Status"])
data["Card_Category"] = le.fit_transform(data["Card_Category"])
data["Income_Category"] = le.fit_transform(data["Income_Category"])

data.head()


# In[323]:


data["Attrition_Flag"].value_counts(normalize = True)


# In[328]:


cor = data.corr()
f, ax = plt.subplots (figsize = (20,20))
sns.heatmap(cor, annot = True, cmap = 'BuGn_r')


# In[333]:


sns.catplot (y = "Months_on_book",kind = "box", data = data, height = 6 )


# In[337]:


data_numerics_only = data.select_dtypes(include=np.number)
data_numerics_only
for column in data_numerics_only.columns:

    sns.distplot(data[column],color="red") # Stack these distributions together with different colors
    plt.show()


# In[341]:


Q1 = data["Months_on_book"].quantile (0.25)
Q3 = data["Months_on_book"].quantile (0.75)

IQR = Q3 - Q1

Upper_Boundry = Q3 + 1.5* IQR
Lower_Boundry = Q1 - 1.5* IQR


# In[343]:


print(Upper_Boundry)
print(Lower_Boundry)


# In[346]:


data = data[(data["Months_on_book"] <= Upper_Boundry) & (data["Months_on_book"]>=Lower_Boundry)]


# In[347]:


sns.catplot (y = "Months_on_book",kind = "box", data = data, height = 6 )


# In[359]:


data.head()


# In[360]:


x = data.drop("Attrition_Flag", axis = 1)
y = data["Attrition_Flag"]


# In[361]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 32)


# In[363]:


x_train, x_test, y_train, y_test.shape


# In[364]:


from sklearn.linear_model import LogisticRegression


# In[365]:


log_reg = LogisticRegression()

log_reg.fit(x_train,y_train)


# In[366]:


y_pred = log_reg.predict(x_test)


# In[368]:


from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score


# In[369]:


accuracy = accuracy_score(y_test,y_pred)
accuracy


# In[370]:


auc = roc_auc_score(y_test, y_pred)
auc


# In[371]:


cm = confusion_matrix(y_pred, y_test)
cm


# In[372]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold


# In[373]:


knn = KNeighborsClassifier()
knn.fit(x_train,y_train)


# In[374]:


y_pred = knn.predict(x_test)


# In[375]:


knn.score(x_train,y_train)


# In[377]:


accuracy_score(y_test,y_pred)


# In[378]:


param_grid = { 'algorithm' : ['ball_tree', 'kd_tree', 'brute'],
               'leaf_size' : [18,20,25,27,30,32,34],
               'n_neighbors' : [3,5,7,9,10,11,12,13]
              }


# In[380]:


from sklearn.model_selection import GridSearchCV


# In[381]:


gridsearch = GridSearchCV(knn, param_grid,verbose=3)


# In[382]:


gridsearch.fit(x_train,y_train)


# In[383]:


gridsearch.best_params_


# In[390]:


knn = KNeighborsClassifier(algorithm ='ball_tree', leaf_size=18, n_neighbors=12)


# In[391]:


knn.fit(x_train,y_train)


# In[392]:


knn.score(x_train,y_train)


# In[393]:


knn.score(x_test,y_test)


# In[394]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()


# In[395]:


model.fit(x_train,y_train)


# In[396]:


y_pred = model.predict(x_test)


# In[397]:


print(accuracy_score(y_test, y_pred))


# In[398]:


conf_mat = confusion_matrix(y_test,y_pred)
conf_mat


# In[399]:


true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]


# In[400]:


Accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
Accuracy


# In[401]:


Precision = true_positive/(true_positive+false_positive)
Precision


# In[402]:


Recall = true_positive/(true_positive+false_negative)
Recall


# In[403]:


F1_Score = 2*(Recall * Precision) / (Recall + Precision)
F1_Score


# In[404]:


auc = roc_auc_score(y_test, y_pred)
auc


# In[405]:


from sklearn.svm import SVC


# In[408]:


model=SVC()
model.fit(x_train,y_train)


# In[409]:


model.predict(x_test)


# In[410]:


accuracy_score(y_test,model.predict(x_test))


# In[431]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.tree import export_graphviz
from IPython.display import Image
get_ipython().system('pip install pydotplus -U')
import pydotplus
get_ipython().system('pip install graphviz')
import graphviz
from sklearn import tree


# In[432]:


clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)


# In[433]:


feature_name=list(X.columns)
class_name = list(y_train.unique())
feature_name


# In[435]:


dot_data = export_graphviz(clf,feature_names = feature_name,rounded = True,filled = True)
# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)  
graph.write_png("myTree.png")
# Show graph
Image(graph.create_png())


# In[436]:


clf.score(x_train,y_train)


# In[438]:


py_pred = clf.predict(x_test)


# In[439]:


clf.score(x_test,y_test)


# Which of the 5 models would you recommend for deployment in the real-world? :- Decision Tress as it has better score
# 
# Is any model underfitting? If yes, what could be the possible reasons? :- Yes as the data is imbalanced 
# 

# In[ ]:




