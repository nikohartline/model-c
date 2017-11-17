#Calculates predictive models for Y1 and Y2 with with logistic regression
#See model-c-data-exploration.R and model-c-cross-validation.py for more details on model selection

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

Modeling_C=pd.read_csv("Modeling_C.csv")

X = Modeling_C[['X1','X2','X3','X4']]
Y1 = Modeling_C.Y1
Y2 = Modeling_C.Y2

#Y1 Model fitting

Y1_model = LogisticRegression()
Y1_model.fit(X[['X1','X4']], Y1)
B0,B1,B4 = np.concatenate((Y1_model.intercept_,Y1_model.coef_[0]),axis=0)
print("Y1 Model: logit(Y1) = %.3f + (%.3f) * X1 + (%.3f) * X4" % (B0,B1,B4))

#Y2 Model fitting

Y2_model = LogisticRegression()
Y2_model.fit(X[['X3']], Y2)
theta0,theta3 = Y2_model.intercept_,Y2_model.coef_[0][0]


print("Y2 Model: logit(Y2) = %.3f + (%.3f) * X3" % (theta0,theta3))
