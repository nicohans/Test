# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 08:23:49 2017

@author: Hans
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.naive_bayes import GaussianNB

def normalized(x):    
    nData = len(x)
    
    #z-score
    aZ = stats.zscore(x,ddof=1)
    
    #normalize
    b = aZ-(np.ones((nData,1))*aZ.min(axis=0))
    c = aZ.max(axis=0) - aZ.min(axis=0)
    x = b/c
    return x

n = pd.read_csv('titanic.csv')
x = n[['Survived','Pclass','Sex','Age','Fare','SibSp','Embarked']]
x = x.fillna(0)

gender = pd.get_dummies(x['Sex'])
embarked = pd.get_dummies(x['Embarked'])

y = x['Survived']
x = pd.concat([x['Pclass'],gender,x['Age'],x['Fare'],x['SibSp'],embarked],axis=1)
x = normalized(x)

gnb = GaussianNB()
gnb = gnb.fit(x,y)

n = pd.read_csv('testTitanic.csv')
x = n[['Survived','Pclass','Sex','Age','Fare','SibSp','Embarked']]
x = x.fillna(0)

gender = pd.get_dummies(x['Sex'])
embarked = pd.get_dummies(x['Embarked'])

y = x['Survived']
x = pd.concat([x['Pclass'],gender,x['Age'],x['Fare'],x['SibSp'],embarked],axis=1)
x = normalized(x)

output = gnb.predict(x)
join = list(zip(y,output))
ok = pd.DataFrame(data = join, columns=['Target','Prediction'])

recrate = float(sum(ok['Target']==ok['Prediction']))/float(len(x))
print recrate

ok.to_csv('outputNB.csv',index=False,header=True)