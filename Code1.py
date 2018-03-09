#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv('Data.csv')

#creating matrix  of features
X=dataset.iloc[:,:-1].values

#creating dependent variable vector
Y=dataset.iloc[:,3].values

#taking care of missing values
from sklearn.preprocessing import Imputer
imputer=Imputer(missing values = 'NaN',strategy = 'mean',axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

