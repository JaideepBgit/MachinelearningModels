#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:24:19 2020

@author: Jaideep Bommidi
"""

"""
Download and install Python Scipy and get the most useful package for machine learning in python

load a dataset and understand it's structure using statistical summaries and data visualization

create a 6 machine learning models, pick the best and get the accuracy realiable

steps:
    Define problem
    prepare data
    evaluate algorithms
    improve results
    present results

Classification of Iris flowers

1. installing python and scipy platform
2. loading the dataset
3. summarizing the dataset
4. visualizing the databset
5. evaluating some algorithms
6. making some predictions
"""

"""
checking for versions of installed libraries
"""

# python version
import sys
print('python: {}'.format(sys.version))

#scipy
import scipy
print('scipy: {}'.format(scipy.__version__))

# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))

# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))

#pandas
import pandas
print('pandas: {}'.format(pandas.__version__))

#scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))


"""
import/load libraries
"""
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

"""
Loading the data
We are going to use Iris dataset
Dataset contains 150 observations.
4 coloumns of measurements of flowers in cms.
fifth column is the species of flower.

Pandas to load and explore the data.
Both descriptive statistics and data visualization
"""

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

"""
Summarize the dataset:
    1. Dimensions of the dataset
    2. peek at the data
    3. Statistical summary of all attributes
    4. Breakdown of data by the class variable
"""

#1 shape
print(dataset.shape)

#2 peek at the data........... prints the first 20 rows of the data
print(dataset.head(20))

#3 descriptions
print(dataset.describe())

#4 class distribution
print(dataset.groupby('class').size())

"""
Data Visualzation
"""

"""
# univariate plots
# box and wisker plots
 univariate to check data distribution of each individual input variables
 given input variables as numeric, create box and wisker plots of each
"""
dataset.plot(kind='box', subplots=True, layout=(2,2),sharex=False,sharey=False)
pyplot.show()


"""
to get an idea of distribution---- we use histogram
2 of the input variables have a gaussian distribution
"""
#histograms
dataset.hist()
pyplot.show()


"""
Multivariate plots - look at the interaction between variables

spot structured relationships between input variables

There is diagonal grouping of pairs of attributes.

It suggests high correlation and a predictable relationship
"""
scatter_matrix(dataset)
pyplot.show()

"""
evaluate some algorithms

1. separate out a validation dataset
2. set-up the test harness to use 10-fold cross validation
3. build multiple different models to predict species from flower measurements
4. select the best model
"""

#split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X,y, test_size=0.20,random_state=1)

"""
6 different algorithms:
    Logistic regression (LR)
    Linear Discriminant Analysis (LDA)
    K-nearest neighbors (KNN)
    Classification and Regression Trees (CART
    Gaussian Naive Bayes (NB)
    Support Vector Machines (SVM)

Good mixture of simple linear (LR and LDA) and nonlinear(KNN, CART, NB, SVM) algos
"""


#spot check algorithms
models = []
models.append((
        'LR', LogisticRegression(solver='liblinear', multi_class='ovr')
        ))
models.append((
        'LDA', LinearDiscriminantAnalysis()
        ))
models.append((
        'KNN', KNeighborsClassifier()
        ))
models.append((
        'CART', DecisionTreeClassifier()
        ))
models.append((
        'NB', GaussianNB()
        ))
models.append((
        'SVM', SVC(gamma='auto')
        ))

# evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)'% (name, cv_results.mean(), cv_results.std()))

# compare algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm comparison')
pyplot.show()































