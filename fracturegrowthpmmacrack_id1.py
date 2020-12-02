#!/usr/bin/env python
# coding: utf-8

# In[69]:


# import relevant commands
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[101]:


# Reading the FractureGrowth dateset Exploring the growth
growth = pd.read_csv('fracturegrowthdata.csv')
growth.head(142)


# In[71]:


growth.describe()


# In[72]:


for i, col in enumerate(['X', 'Y']):
    plt.figure(i)
    sns.catplot(x=col, y='Fracture Growth', data=growth, kind='point', aspect=2,)


# In[73]:


from sklearn.model_selection import train_test_split


# In[74]:


features = growth.drop('Fracture Growth', axis=1)
labels = growth['Fracture Growth']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


# In[75]:


for dataset in(y_train, y_val, y_test):
    print(round(len(dataset) / len(labels), 2))


# In[76]:


print(len(labels), len(y_train), len(y_val), len(y_test))


# In[77]:


# Write out train, test, validation data
X_train.to_csv('Fracture_train_features.csv' , index=False)
X_val.to_csv('Fracture_val_features.csv' , index=False)
X_test.to_csv('Fracture_test_features.csv' , index=False)

y_train.to_csv('Fracture_train_labels.csv' , index=False)
y_val.to_csv('Fracture_val_labels.csv' , index=False)
y_test.to_csv('Fracture_test_labels.csv' , index=False)


# In[78]:


# Logistic Regression Model
# Gives potential hyperparameters we could tune. We would tune the one which has the largest impact
# The C hyperparameter is regularisation parameter is logistic reg that controls how closely model fits to training data
# More value of C, less regularization and classification good, if less C then more regularization and underfitting
from sklearn.linear_model import LogisticRegression

LogisticRegression()


# In[90]:


import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

tr_features = pd.read_csv('Fracture_train_features.csv')
tr_labels = pd.read_csv('Fracture_train_labels.csv', header=None)


# In[91]:


def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))
    
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean,3), round(std * 2, 3), params))


# In[92]:


# Logistic Regression with 4-fold cross validation and tuning the hyperparameter C
# Shows that model underfits when low C & high Regularization & less accuracy; C=1 best accuracy; model overfits when high C & low Reg
# 7 hyperparameters and 10 cross val so 7*10 = 70 individual models
# Best is 78% as below
lr = LogisticRegression()
parameters = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
}

cv = GridSearchCV(lr, parameters, cv=10)
cv.fit(tr_features, tr_labels.values.ravel())

print_results(cv)


# In[93]:


cv.best_estimator_


# In[94]:


# Write out pickled model
joblib.dump(cv.best_estimator_,'FGLR_model.pkl')


# In[95]:


# Evaluate results on the validation set
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from time import time

val_features = pd.read_csv('Fracture_val_features.csv')
val_labels = pd.read_csv('Fracture_val_labels.csv', header=None)

te_features = pd.read_csv('Fracture_test_features.csv')
te_labels = pd.read_csv('Fracture_test_labels.csv', header=None)


# In[96]:


models = {}

for mdl in ['FGLR']:
    models[mdl] = joblib.load('{}_model.pkl'.format(mdl))


# In[97]:


models


# In[98]:


# Evaluate models on the validation set
def evaluate_model(name, model, features, labels):
    start = time()
    pred = model.predict(features)
    end = time()
    accuracy = round(accuracy_score(labels, pred), 3)
    precision = round(precision_score(labels, pred), 3)
    recall = round(recall_score(labels, pred), 3)
    print('{} -- Accuracy: {} / Precision: {} / Recall: {} / Latency: {}ms'.format(name, accuracy, precision, recall, round((end - start))))


# In[99]:


for name, mdl in models.items():
    evaluate_model(name, mdl, val_features, val_labels)


# In[100]:


for name, mdl in models.items():
    evaluate_model(name, mdl, te_features, te_labels)


# In[ ]:




