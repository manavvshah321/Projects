#Importing heart.csv
heart = read.csv("/Users/divyeshshah/Desktop/ISYE 312/heart.csv")

#Scatter plots of Resting Blood Pressure vs Age, Max Heart Rate, Cholestrol
plot(data$age, data$trestbps, xlab = "Age", ylab = "Resting Blood Pressure")
plot(data$chol, data$trestbps, xlab = "Cholesterol", ylab = "Resting Blood Pressure")
plot(data$thalach, data$trestbps, xlab = "Maximum Heart Rate", ylab = "Resting Blood Pressure")

#Boxplots of Resting Blood Pressure vs Chest Pain Type, Sex, Fasting Blood Pressure
boxplot(data$trestbps~data$sex, names = genders, xlab = "Sex", ylab = "Resting Blood Pressure")
boxplot(data$trestbps~data$cp, names = cp_type, xlab = "Chest Pain Type", ylab = "Resting Blood Pressure")
boxplot(data$trestbps~data$fbs, names = conditions, xlab = "Fasting Blood Pressure > 120 mg/dl", ylab = "Resting Blood Pressure")

#Simple Linear Regression of Resting Blood Pressure on Age
simple_model = lm(heart$trestbps~heart$age)
summary(simple_model)

#Multiple Linear Regression 
multiple_model = lm(heart$trestbps~heart$age+heart$chol+heart$thalach+heart$sex+heart$cp+heart$fbs)
summary(multiple_model)

#Brand Preference Scatterplot Matrix
pairs(~heart$trestbps+heart$age+heart$chol+heart$thalach+heart$sex+heart$cp+heart$fbs,main="Scatter plot matrix")

#Variance Inflation Factors
car::vif(multiple_model)

#Residuals vs Fitted Plot
yhat = multiple_model$fitted.values
res = multiple_model$residuals
plot(yhat, res, ylab="Residuals", xlab="Fitted Value", main="Residual vs Fitted")
abline(0,0)

#Cook's Distance
print(influence.measures(multiple_model))

#Residual Q-Q Plot
qqnorm(res, ylab = "Residuals",main = "Residual Q-Q Plot")
qqline(res)

#Interaction Model
interaction_model = lm(heart$trestbps~heart$age+heart$chol+heart$thalach+heart$sex+heart$cp+heart$fbs+heart$chol*heart$age)

#Anova of Interaction Model
anova(multiple_model, interaction_model)

#Residual Plot for Alternate Model
new_res = interaction_model$residuals
new_fitted = interaction_model$fitted.values
plot(new_fitted, new_res)

#Enhanced Regression
library(car)
qqPlot(multiple_model)

#Linear Absolute Deviation
library(quantreg)
lad_model = rq(heart$trestbps~heart$age+heart$chol+heart$thalach+heart$sex+heart$cp+heart$fbs)
lad_model






