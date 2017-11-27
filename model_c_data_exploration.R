#### Exploratory Analysis of WattTime Model Data  ####

require(data.table)
require(dplyr)
require(ggplot2)

Modeling_C = fread("Modeling_C.csv")

possible_outlier_rows = c(which(Modeling_C$X1 > 300),
                          which(Modeling_C$X1 < -50 & Modeling_C$X4 < -20)) #Labels two outlier rows. Neither has much impact on model assumptions or coefficients herein, but they are kept in the model nonetheless. Uncomment line below to remove them for analyses. They are
# Modeling_C=Modeling_C[-possible_outlier_rows,]

#Boxplots of each explanatory variable to visualize distribution
windows()
par(mfrow = c(2,2))
boxplot(Modeling_C$X1); title("X1")
boxplot(Modeling_C$X2); title("X2")
boxplot(Modeling_C$X3); title("X3")
boxplot(Modeling_C$X4); title("X4")

cor(Modeling_C, method="pearson") #Correlation matrix of variables
cor.test(Modeling_C$X1, Modeling_C$X3, method = "pearson") #correlation coefficient of 0.9 (p < 0.001). Introduces problems of multicollinearity in models that include both the X1 and X3 variables. X1 is used as the predictor for Y1 and X3 is used as the predictor for Y2.

#### Model Selection for Y1 ####

cbrt = function(x) {
  sign(x) * abs(x)^(1/3)
}

Modeling_C = mutate(Modeling_C,
                    X1_sq=X1^2,
                    X1_cb=X1^3,
                    X2_sq=X2^2,
                    X2_cb=X2^3,
                    X3_sq=X3^2,
                    X3_cb=X3^3,
                    X4_sq=X4^2,
                    X4_cb=X4^3,
                    cbrtX1=cbrt(X1),
                    cbrtX2=cbrt(X2),
                    cbrtX3=cbrt(X3),
                    cbrtX4=cbrt(X4)) #creation of features for model selection

y1_logit_model_initial = glm("Y1~X1+X2+X4+X1_sq+X2_sq+X4_sq",
                             family = "binomial",
                             data = Modeling_C,
                             control = list(maxit = 100))

summary(y1_logit_model_initial)
#The estimators for X2 and X2_sq had an insignificant effect on the outcome and are excluded from further models. X1_sq and X4_sq had significant coefficients, but are excluded from the model for interpretability (small impact on predictive capacity of model).

y1_logit_model_x1_x4 = glm("Y1~X1+X4",
                           family = "binomial",
                           data = Modeling_C,
                           control = list(maxit = 100))

summary(y1_logit_model_x1_x4)
#Chosen Model#

y1_logit_model_x1_cbrtx4 = glm("Y1~X1+cbrtX4",
                           family = "binomial",
                           data = Modeling_C,
                           control = list(maxit = 100))

summary(y1_logit_model_x1_cbrtx4)


data_scatter_y1 = ggplot(data = Modeling_C[-possible_outlier_rows,],
                         aes(x = X1, y = cbrtX4, color = Y1))+
  geom_point()
windows()
data_scatter_y1 #simple plot showing the distribution of X1 and X4 values with corresponding Y1 values encoded as light blue (Y1 = 1) or dark blue (Y1 = 0)

#### Model Selection for Y2 ####

#including both X3_sq and X3 causes the model to be unable to converge (X3_sq is removed)
y2_logit_model_poly = glm("Y2~X2+X3+X4+X2_sq+X4_sq",
                          family = "binomial",
                          data = Modeling_C,
                          control = list(maxit = 100))

summary(y2_logit_model_poly)
#X2_sq and X4_sq are removed from the model

y2_logit_model_interaction = glm("Y2~X2+X3+X4+X2*X3+X2*X4+X3*X4",
                                 family = "binomial",
                                 data = Modeling_C,
                                 control = list(maxit = 100))

summary(y2_logit_model_interaction)
#X2*X4 are removed from the model

y2_logit_model_x2_x3_x4_x2.x3_x3.x4 = glm("Y2~X2+X3+X4+X2*X3+X3*X4",
                                          family = "binomial",
                                          data = Modeling_C,
                                          control = list(maxit = 100))

summary(y2_logit_model_x2_x3_x4_x2.x3_x3.x4)
#This model contains all significant terms, but X3 is by far the strongest predictor. For interpretability, the chosen model excludes all variables but X3


y2_logit_model_x1_x4 = glm("Y2~X3",
                           family = "binomial",
                           data = Modeling_C,
                           control = list(maxit = 100))

summary(y1_logit_model_x1_x4)
#Chosen Model#

data_scatter_y2 = ggplot(data = Modeling_C[-possible_outlier_rows,],
                         aes(x = X3,y = Y2,color = Y2))+
  geom_point()
windows()
data_scatter_y2
