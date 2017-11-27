#Calculates predictive models for Y1 and Y2 with with logistic regression
#See model-c-data-exploration.R and model-c-cross-validation.py for more details on model selection

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

Modeling_C=pd.read_csv("Modeling_C.csv")

Y1 = Modeling_C.Y1
Y2 = Modeling_C.Y2

from sklearn.preprocessing import PolynomialFeatures
poly3 = PolynomialFeatures(3) #create polynomial and interaction features with data

X = poly3.fit_transform(Modeling_C[['X1','X2','X3','X4']])[:,1:]

feature_names = poly3.get_feature_names(['X1','X2','X3','X4'])[1:] + ['cbrtX1','cbrtX2','cbrtX3','cbrtX4']

from math import copysign
def cbrt(x):
    """function to take cube root of columns with negative data without producing NaN values"""
    return copysign(abs(x)**(1/3),x)
cbrt = np.vectorize(cbrt)

X = np.append(X, cbrt(X[:,0:4]), axis=1) #add the cube root features

X = pd.DataFrame(X,columns = feature_names)

#Y1 Model fitting

Y1_model = LogisticRegression()
Y1_model.fit(X[['X1','cbrtX4']], Y1)
B0,B1,B4 = np.concatenate((Y1_model.intercept_,Y1_model.coef_[0]),axis=0)
print("Y1 Model: logit(Y1) = %.3f + (%.3f) * X1 + (%.3f) * X4^(1/3)" % (B0,B1,B4))

#Y2 Model fitting

Y2_model = LogisticRegression()
Y2_model.fit(X[['X3']], Y2)
theta0,theta3 = Y2_model.intercept_,Y2_model.coef_[0][0]


print("Y2 Model: logit(Y2) = %.3f + (%.3f) * X3" % (theta0,theta3))
