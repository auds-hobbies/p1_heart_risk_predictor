# HEART RISK PREDICTOR APP
"""
For this project, I developed an Excel dashboard, a Power BI dashboard, and a Django web app.
Remaining files can be found on github
"""

# LOAD LIBRARIES
# #Indeed, loaded more than necessary packages!!
import pandas as pd
from collections import Counter
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
from numpy import mean, std, NaN
from stop_words import get_stop_words
import scipy.stats as stats
import random
#from random import seed
#seed(1)

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot
import dash
import dash_table_experiments as dt
from plotly import graph_objects as go
from plotly.graph_objs import *
from dash.dependencies import Input, Output, State
import plotly.express as px

from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline #as imbPipeline

from joblib import dump
from joblib import load
import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.decomposition import NMF
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, FunctionTransformer, PowerTransformer, RobustScaler,MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier , GradientBoostingClassifier,RandomForestClassifier,ExtraTreesClassifier,RandomForestRegressor

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import classification_report, confusion_matrix, auc, accuracy_score,log_loss,roc_auc_score,roc_curve, make_scorer,r2_score,jaccard_score
from imblearn.metrics import geometric_mean_score


pd.options.display.max_columns = 35
pd.options.display.float_format ='{:.2f}'.format

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# LOAD FILE
file_heart = "heart.csv"

df00 = pd.read_csv(file_heart, delimiter=',')
print(df00.shape)
df00.head()

# IDENTIFY & FILL MISSING VALUES
missing_cols = [col for col in df00.columns if df00[col].isna().sum()>0]
print(f"Columns with missing values = {missing_cols}\n")

df00['age'].fillna(df00['age'].median(), inplace=True)
df00['sex'].fillna(df00['sex'].median(), inplace=True)
df00['cp'].fillna(df00['cp'].median(), inplace=True)
df00['trestbps'].fillna(df00['trestbps'].median(), inplace=True)
df00['chol'].fillna(df00['chol'].median(), inplace=True)
df00['fbs'].fillna(df00['fbs'].median(), inplace=True)
df00['restecg'].fillna(df00['restecg'].median(), inplace=True)
df00['thalach'].fillna(df00['thalach'].median(), inplace=True)
df00['exang'].fillna(df00['exang'].median(), inplace=True)
df00['oldpeak'].fillna(df00['oldpeak'].median(), inplace=True)
df00['slope'].fillna(df00['slope'].median(), inplace=True)
df00['ca'].fillna(df00['ca'].median(), inplace=True)
df00['thal'].fillna(df00['thal'].median(), inplace=True)
df00['target'].fillna(df00['target'].median(), inplace=True)


df00.target = df00.target.astype('O')
df00.sex = df00.sex.astype('O')
df00.cp = df00.cp.astype('O')
df00.fbs = df00.fbs.astype('O')
df00.restecg = df00.restecg.astype('O')
df00.exang = df00.exang.astype('O')
df00.slope = df00.slope.astype('O')
df00.ca = df00.ca.astype('O')
df00.thal = df00.thal.astype('O')

df00.info()

# IDENTIFY FEATURE TYPES
df = df00.copy()

discrete_type = ['int16', 'int32', 'int64']
continuous_type =['float16', 'float32', 'float64']


discrete = [var for var in df.columns if df[var].dtype !='O' and df[var].dtype in discrete_type and df[var].nunique() < df.shape[0]]
continuous = [var for var in df.columns if df[var].dtype !='O' and df[var].dtype in continuous_type and var not in discrete]

combo = [discrete, continuous]
categorical = [var for var in df.columns if df[var].dtype=='O' and var not in combo and df[var].nunique()!=df.shape[0]]

string_all = [discrete, continuous, categorical]
unique_identifier = [var for var in df.columns if df[var].dtype=='O' and var not in string_all and df[var].nunique()==len(df)]


print(f'Total number of variables = {df.shape[1]}\n')
print('# of Unique IDs = {}'.format(len(unique_identifier)))
print('# of discrete variables = {}'.format(len(discrete)))
print('# of continuous variables = {}'.format(len(continuous)))
print('# of categorical variables = {}\n'.format(len(categorical)))
#print('There are {} mixed variables'.format(len(mixed)))
print(f'Categorical variables = {categorical}\n')
print(f'Continuous variables = {continuous}\n')
print(f'Discrete variables = {discrete}\n')

# FEATURE ENGINEERING
def age_classify(x):
    score = x['age']
    if score in range(0, 46):
        return 'age_upto45'
    elif score in range(46, 61):
        return 'age_46to60'
    elif score in range(61, 120):
        return 'age_above60'
    else:
        return 'unknown_age'


df['age_group'] = df.apply(age_classify, axis=1)


def trestbps_classify(x):
    score = x['trestbps']
    if score in range(0, 120):
        return 'trestbps_upto120'
    elif score in range(120, 160):
        return 'trestbps_121to160'
    elif score >= 160:
        return 'trestbps_above160'
    else:  # score < 0:
        return 'trestbps_below0'


df['trestbps_group'] = df.apply(trestbps_classify, axis=1)


def chol_classify(x):
    score = x['chol']
    if score in range(0, 250):
        return 'chol_upto250'
    elif score >= 250:
        return 'chol_above250'
    else:  # score < 0:
        return 'chol_unknown0'


df['chol_group'] = df.apply(chol_classify, axis=1)


def thalach_classify(x):
    score = x['thalach']
    if score in range(0, 140):
        return 'thalach_upto140'
    elif score in range(140, 160):
        return 'thalach_140to160'
    elif score >= 160:
        return 'thalach_above160'
    else:  # score < 0:
        return 'thalach_unknown0'


df['thalach_group'] = df.apply(thalach_classify, axis=1)

df.thalach_group.value_counts()

# TRAIN TEST SPLIT
#Re-run
discrete = [var for var in df.columns if df[var].dtype !='O' and df[var].dtype in discrete_type and df[var].nunique() < df.shape[0]]
continuous = [var for var in df.columns if df[var].dtype !='O' and df[var].dtype in continuous_type and var not in discrete]

combo = [discrete, continuous]
categorical = [var for var in df.columns if df[var].dtype=='O' and var not in combo and df[var].nunique()!=df.shape[0]]



""" 
['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target', 'age_group',
       'trestbps_group', 'chol_group', 'thalach_group']
"""
pre_X = df.drop(columns=['target','age','trestbps','chol','thalach'], axis=1)
pre_y = df['target']

X_train,X_test,y_train,y_test= train_test_split(pre_X, pre_y, stratify=pre_y, test_size=0.3, random_state=0)

X_train.shape, X_test.shape


# TRANSFORMS - LABEL ENCODING
def le_transform(xtrain, ytrain, xtest, ytest):
    cat_inputs_only = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'age_group', 'trestbps_group',
                       'chol_group',
                       'thalach_group']

    le = LabelEncoder()
    X_train_le = xtrain.copy()
    X_test_le = xtest.copy()
    y_train_le = ytrain.copy()
    y_test_le = ytest.copy()
    y_train_le = y_train_le.replace(('0', '1'), (0, 1))  # , inplace=True )
    y_test_le = y_test_le.replace(('0', '1'), (0, 1))  # , inplace=True )

    for feat in cat_inputs_only:
        X_train_le[feat] = le.fit_transform(X_train_le[feat].astype(str))
        X_test_le[feat] = le.fit_transform(X_test_le[feat].astype(str))

    return X_train_le, y_train_le, X_test_le, y_test_le


X_train_le, y_train_le, X_test_le, y_test_le = le_transform(X_train, y_train, X_test, y_test)

# PREDICTIVE MODELLING
randomforest = RandomForestClassifier()
randomforest.fit(X_train_le, y_train_le)
y_pred = randomforest.predict(X_test_le)
acc_randomforest = round(accuracy_score(y_pred, y_test_le) * 100, 2)
print(f"Random Forest Acc = {acc_randomforest}")


#PICKLE THE MODEL
pickle.dump(randomforest, open('heart_risk_predictor.sav','wb'))


def prediction_model(sex,cp,fbs,restecg,exang,oldpeak,slope,ca,thal,age_group,trestbps_group,chol_group,thalach_group):
    import pickle
    x = [[sex,cp,fbs,restecg,exang,oldpeak,slope,ca,thal,age_group,trestbps_group,chol_group,thalach_group]]
    randomforest2 = pickle.load(open('heart_risk_predictor.sav','rb'))
    predictions2 = randomforest2.predict(x)
    print(predictions2)

prediction_model(1, 3, 0, 2, 0, 105.7, 0, 4, 3, 2, 0, 1, 0 )
