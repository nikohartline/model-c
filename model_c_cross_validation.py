import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

Modeling_C=pd.read_csv("Modeling_C.csv")
Modeling_C=Modeling_C.sample(frac=1) #randomize rows for model training

corr_matrix = Modeling_C.corr()
X1_X3_corr_test=scipy.stats.pearsonr(Modeling_C.X1,Modeling_C.X3) #Calculates the correlation and p-value (finding significant (p<0.001) correlation (0.9) between X1 and X3)

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

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

LR_model = LogisticRegression()
SVC_model_test = SVC()

cv_scores_Y1_X1_X3_X4 = cross_val_score(LR_model, X[['X1','X3','X4']], Y1, cv=10)
cv_scores_Y1_X1_X4 = cross_val_score(LR_model, X[['X1','X4']], Y1, cv=10)
cv_scores_Y1_X1_cbrtX4 = cross_val_score(LR_model, X[['X1','cbrtX4']], Y1, cv=10)
cv_scores_Y1_X1_X1sq_X4 = cross_val_score(LR_model, X[['X1','X1^2','X4']], Y1, cv=10)
#cv_scores_SVC = cross_val_score(SVC_model_test, X, Y1, cv=10)

print("Average accuracy of 10-fold cross-validation for logit model with X1, X3, and X4: %.1f%%" % np.average(cv_scores_Y1_X1_X3_X4*100))
print("Average accuracy of 10-fold cross-validation for logit model with X1 and X4: %.1f%%" % np.average(cv_scores_Y1_X1_X4*100))
print("Average accuracy of 10-fold cross-validation for logit model with X1 and X4^(1/3): %.1f%%" % np.average(cv_scores_Y1_X1_cbrtX4*100))
print("Average accuracy of 10-fold cross-validation for logit model with X1, X1^2, and X4: %.1f%%" % np.average(cv_scores_Y1_X1_X1sq_X4*100))
#print("Average accuracy of 10-fold cross-validation for support vector classification (SVC) model: %.1f%%" % np.average(cv_scores_SVC*100))

LR_model.fit(X[['X1','cbrtX4']], Y1)

print("Y1 Model Selected: logit(Y1) = %.3f + (%.3f) * X1 + (%.3f) * X4^(1/3)" % (LR_model.intercept_,LR_model.coef_[0][0],LR_model.coef_[0][1]))

y1_cm = confusion_matrix(Y1,LR_model.predict(X[['X1','X4']]))
y1_cm = pd.DataFrame(y1_cm,columns=["0","1"],index=["0","1"])
print("Confusion matrix (rows are actual and columns are predicted)")
print(y1_cm)

cv_scores_Y2_X2_X3_X4 = cross_val_score(LR_model, X[['X2','X3','X4']], Y2, cv=10)
cv_scores_Y2_X3_X4 = cross_val_score(LR_model, X[['X3','X4']], Y2, cv=10)
cv_scores_Y2_X2_X3_X4_X2X3_X3X4 = cross_val_score(LR_model, X[['X2','X3','X4','X2 X3','X3 X4']], Y2, cv=10)
cv_scores_Y2_X3_X4_X3X4 = cross_val_score(LR_model, X[['X3','X4','X3 X4']], Y2, cv=10)
cv_scores_Y2_X3 = cross_val_score(LR_model, X[['X3']], Y2, cv=10)

print("Average accuracy of 10-fold cross-validation for logit model with X1, X3, and X4: %.1f%%" % np.average(cv_scores_Y2_X2_X3_X4*100))
print("Average accuracy of 10-fold cross-validation for logit model with X3 and X4: %.1f%%" % np.average(cv_scores_Y2_X3_X4*100))
print("Average accuracy of 10-fold cross-validation for logit model with X2, X3, X4, X2:X3, and X3:X4: %.1f%%" % np.average(cv_scores_Y2_X2_X3_X4_X2X3_X3X4*100))
print("Average accuracy of 10-fold cross-validation for logit model with X3, X4, and X3:X4: %.1f%%" % np.average(cv_scores_Y2_X3_X4_X3X4*100))
print("Average accuracy of 10-fold cross-validation for logit model with X3: %.1f%%" % np.average(cv_scores_Y2_X3*100))

LR_model.fit(X[['X3']], Y2)

print("Y2 Model Selected: logit(Y2) = %.3f + (%.3f) * X3" % (LR_model.intercept_,LR_model.coef_[0][0]))

y2_cm = confusion_matrix(Y2,LR_model.predict(X[['X3']]))
y2_cm = pd.DataFrame(y2_cm,columns=["0","1"],index=["0","1"])
print("Confusion Matrix (rows are actual and columns are predicted)")
print(y2_cm)
