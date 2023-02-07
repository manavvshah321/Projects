######## DS project - Vehicle Insurance #######
source("DataAnalyticsFunctions.R")
installpkg("tree")
library(tree)
installpkg("partykit")
installpkg("libcoin")
installpkg("randomForest")
library(randomForest)
library(libcoin)
library(partykit)
installpkg("glmnet")
library(glmnet)
library(ggplot2)
library(corrplot)
### This will turn off warning messages
options(warn=-1)

set.seed(1)

#Importing the data into R
vehicle_insurance <- read.csv("Vehicle Insurance.csv")
View(vehicle_insurance)

#Initial Review of the Data
summary(vehicle_insurance) #Check for NA values - none noted
str(vehicle_insurance)

#Creating Dummy Variables for Categorical Variables
vehicle_age_greater_than_2_years <- ifelse(vehicle_insurance$Vehicle_Age == "> 2 Years", 1, 0)
vehicle_age_between_1_and_2_years <- ifelse(vehicle_insurance$Vehicle_Age == "1-2 Year", 1, 0)
vehicle_age_less_than_1_year <- ifelse(vehicle_insurance$Vehicle_Age == "< 1 Year", 1, 0)
Gender01 <- ifelse(vehicle_insurance$Gender == "Male", 1, 0)
vehicle_damage <- ifelse(vehicle_insurance$Vehicle_Damage == "Yes", 1, 0)

vehicle_insurance <- data.frame(vehicle_insurance, vehicle_age_greater_than_2_years, vehicle_age_between_1_and_2_years, vehicle_age_less_than_1_year, Gender01, vehicle_damage)
vehicle_insurance <- subset(vehicle_insurance, select = -c(Gender, Vehicle_Age, Vehicle_Damage, id))

### Data Understanding and Visualizations ###

##Distribution of age, count and vehicle damage



##Correlation Matrix Plot 
M<-cor(vehicle_insurance)
M

rownames(M) <- c("Age", "Driving License","Region Code", "Previously Insured","Annual Premium", "Policy Sales Channel","Vintage", "Response", "Vehicle Age > 2 Years", "1 Year <= Vehicle Age <= 2 Years", "Vehicle Age < 1 Year", "Gender", "Vehicle Damage")
colnames(M) <- c("Age", "Driving License","Region Code", "Previously Insured","Annual Premium", "Policy Sales Channel","Vintage", "Response", "Vehicle Age > 2 Years", "1 Year <= Vehicle Age <= 2 Years", "Vehicle Age < 1 Year", "Gender", "Vehicle Damage")
corrplot(M,method="square", tl.col = "black") 


##Grouping customers by response and previous insurance
m <- aggregate(Response==1 ~ Previously_Insured + vehicle_damage, data=vehicle_insurance, FUN = sum)
Labelsm <- paste(m[,1],"/",m[,2])
Labelsm[3] <- "\n No Vehicle Insurance \n Had Vehicle Damage"
Labelsm[1] <- "\n No Vehicle Insurance \n No Vehicle Damage"
Labelsm[2] <- "\n Has Vehicle Insurance \n No Vehicle Damage"
Labelsm[4] <- "\n Has Vehicle Insurance \n Had Vehicle Damage"
bp = barplot( m[,3], names.arg = Labelsm, col=c(5), ylab = "Count of Interested Customers", xlab = "Previously Insured/Vehicle Damage", main = "Interested Customers Based on \n Vehicle Insurance and Vehicle Damage", ylim= c(0,52000), density = c(60))
text(x = bp, y =  m[,3], label = m[,3], pos = 3, cex = 0.8, col = "black" )

#Age distribution of customers that have had vehicle damage in the past
vehicle_response_df <- vehicle_insurance[vehicle_insurance$Response == 1, ]
vehicle_response_df
ggplot(vehicle_response_df, aes(x=Age)) + ylab("Count of Vehicle Damage") + xlab("Age of Each Customer") + geom_histogram(binwidth=.5, color = "purple") + ggtitle("Age Distribution of Customers Interested in Vehicle Insurance")


mean(vehicle_insurance$Response==1) #this shows that the dataset is imbalanced as only 12.2% of customers are interested in vehicle insurance

#Creating initial models
model.logistic.interaction <-glm(Response==1~.^2, data=vehicle_insurance, family="binomial")
model.logistic <-glm(Response==1~., data=vehicle_insurance, family="binomial")
model.tree <- tree(factor(Response)~., data=vehicle_insurance) 

#Defining appropriate variables for Lasso and Post Lasso Models
Mx<- model.matrix(Response ~ .^2, data=vehicle_insurance)[,-1] 
My<- vehicle_insurance$Response == 1 

num.features <- ncol(Mx) 
num.n <- nrow(Mx) 
num.resp <- sum(My) 
w <- (num.resp/num.n)*(1-(num.resp/num.n)) 
#### For the binomial case, a theoretically valid choice is
lambda.theory <- sqrt(w*log(num.features/0.05)/num.n) 
lassoCV <- cv.glmnet(Mx,My, family="binomial")
lambda.min <- lassoCV$lambda.min
lambda.1se <- lassoCV$lambda.1se

lasso <- glmnet(Mx,My, family="binomial")
lassoTheory <- glmnet(Mx,My, family="binomial",lambda = lambda.theory, alpha = 1) 
lassoMin <- glmnet(Mx,My, family="binomial",lambda = lambda.min, alpha = 1) 
lasso1se <- glmnet(Mx,My, family="binomial",lambda = lambda.1se, alpha = 1) 

#Creating tables to support with cross validation
features.min <- support(lasso$beta[,which.min(lassoCV$cvm)]) 
features.1se <- support(lasso$beta[,which.min( (lassoCV$lambda-lassoCV$lambda.1se)^2)])
features.theory <- support(lassoTheory$beta)

data.min <- data.frame(Mx[,features.min],My)
data.1se <- data.frame(Mx[,features.1se],My)
data.theory <- data.frame(Mx[,features.theory],My) 


### K Fold Cross Validation with 10 folds
###
n <- nrow(vehicle_insurance)
### create a vector of fold memberships (random order)
nfold <- 10
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]
### create an empty dataframe of results
OOS.R2 <- data.frame(logistic.interaction=rep(NA,nfold), logistic=rep(NA,nfold), tree=rep(NA,nfold), L.min=rep(NA,nfold), L.1se=rep(NA,nfold), L.theory=rep(NA,nfold), PL.min=rep(NA,nfold), PL.1se=rep(NA,nfold), PL.theory=rep(NA,nfold), null=rep(NA,nfold)) 

OOS.Ac <- data.frame(logistic.interaction=rep(NA,nfold), logistic=rep(NA,nfold), tree=rep(NA,nfold), L.min=rep(NA,nfold), L.1se=rep(NA,nfold), L.theory=rep(NA,nfold), PL.min=rep(NA,nfold), PL.1se=rep(NA,nfold), PL.theory=rep(NA,nfold) ,null=rep(NA,nfold)) 
OOS.TP <- data.frame(logistic.interaction=rep(NA,nfold), logistic=rep(NA,nfold), tree=rep(NA,nfold), L.min=rep(NA,nfold), L.1se=rep(NA,nfold), L.theory=rep(NA,nfold), PL.min=rep(NA,nfold), PL.1se=rep(NA,nfold), PL.theory=rep(NA,nfold) ,null=rep(NA,nfold)) 
OOS.TN <- data.frame(logistic.interaction=rep(NA,nfold), logistic=rep(NA,nfold), tree=rep(NA,nfold), L.min=rep(NA,nfold), L.1se=rep(NA,nfold), L.theory=rep(NA,nfold), PL.min=rep(NA,nfold), PL.1se=rep(NA,nfold), PL.theory=rep(NA,nfold) ,null=rep(NA,nfold)) 
OOS.FP <- data.frame(logistic.interaction=rep(NA,nfold), logistic=rep(NA,nfold), tree=rep(NA,nfold), L.min=rep(NA,nfold), L.1se=rep(NA,nfold), L.theory=rep(NA,nfold), PL.min=rep(NA,nfold), PL.1se=rep(NA,nfold), PL.theory=rep(NA,nfold) ,null=rep(NA,nfold)) 
OOS.FN <- data.frame(logistic.interaction=rep(NA,nfold), logistic=rep(NA,nfold), tree=rep(NA,nfold), L.min=rep(NA,nfold), L.1se=rep(NA,nfold), L.theory=rep(NA,nfold), PL.min=rep(NA,nfold), PL.1se=rep(NA,nfold), PL.theory=rep(NA,nfold) ,null=rep(NA,nfold)) 

val <- .2 # due to assymetric costs and benefits associated with advertising


### Use a for loop to run through the nfold trails
for(k in 1:nfold){ 
  train <- which(foldid!=k) # train on all but fold `k'
  
  model.logistic.interaction <-glm(Response==1~.^2, data=vehicle_insurance, subset=train, family="binomial")
  model.logistic <-glm(Response==1~., data=vehicle_insurance, subset=train,family="binomial")
  model.tree <- tree(factor(Response)~., data=vehicle_insurance, subset=train) 
  
  lassomin  <- glmnet(Mx[train,],My[train], family="binomial",lambda = lassoCV$lambda.min)
  lasso1se  <- glmnet(Mx[train,],My[train], family="binomial",lambda = lassoCV$lambda.1se)
  lassoTheory <- glmnet(Mx[train,],My[train], family="binomial",lambda = lambda.theory)
  
  rmin <- glm(My~., data=data.min, subset=train, family="binomial")
  r1se <- glm(My~., data=data.1se, subset=train, family="binomial")
  rtheory <- glm(My~., data=data.theory, subset=train, family="binomial")
  
  model.nulll <-glm(Response==1~1, data=vehicle_insurance, subset=train,family="binomial")
  
  ## get predictions: type=response so we have probabilities
  pred.logistic.interaction <- predict(model.logistic.interaction, newdata=vehicle_insurance[-train,], type="response")
  pred.logistic             <- predict(model.logistic, newdata=vehicle_insurance[-train,], type="response")
  pred.tree                 <- predict(model.tree, newdata=vehicle_insurance[-train,], type="vector")
  pred.tree <- pred.tree[,2]
  
  predlassomin <- predict(lassomin, newx=Mx[-train,], type="response")
  predlasso1se  <- predict(lasso1se, newx=Mx[-train,], type="response")
  predlassotheory <- predict(lassoTheory, newx=Mx[-train,], type="response")
  
  predmin <- predict(rmin, newdata=data.min[-train,], type="response")
  pred1se  <- predict(r1se, newdata=data.1se[-train,], type="response")
  predtheory <- predict(rtheory, newdata=data.theory[-train,], type="response")
  
  pred.null <- predict(model.nulll, newdata=vehicle_insurance[-train,], type="response")
  
  ## calculate and log R2
  # Logistic Interaction
  OOS.R2$logistic.interaction[k] <- R2(y=vehicle_insurance$Response[-train]==1, pred=pred.logistic.interaction, family="binomial")
  # Logistic
  OOS.R2$logistic[k] <- R2(y=vehicle_insurance$Response[-train]==1, pred=pred.logistic, family="binomial")
  # Tree
  OOS.R2$tree[k] <- R2(y=vehicle_insurance$Response[-train]==1, pred=pred.tree, family="binomial")
  #Lasso Min
  OOS.R2$L.min[k] <- R2(y=My[-train], pred=predlassomin, family="binomial")
  #Lasso 1 se
  OOS.R2$L.1se[k] <- R2(y=My[-train], pred=predlasso1se, family="binomial")
  #Lasso theory
  OOS.R2$L.theory[k] <- R2(y=My[-train], pred=predlassotheory, family="binomial")
  #Post Lasso Min
  OOS.R2$PL.min[k] <- R2(y=My[-train], pred=predmin, family="binomial")
  #Post Lasso 1 se
  OOS.R2$PL.1se[k] <- R2(y=My[-train], pred=pred1se, family="binomial")
  #Post Lasso theory
  OOS.R2$PL.theory[k] <- R2(y=My[-train], pred=predtheory, family="binomial")
  
  #Null
  OOS.R2$null[k] <- R2(y=vehicle_insurance$Response[-train]==1, pred=pred.null, family="binomial")
  OOS.R2$null[k]
  #Null Model guess
  sum(vehicle_insurance$Response[train]==1)/length(train)
  
  #Accuracy Predictions 
  #Post Lasso
  values <- FPR_TPR( (predmin >= val) , My[-train] )
  OOS.Ac$PL.min[k] <- values$ACC
  OOS.TP$PL.min[k] <- values$TP
  OOS.TN$PL.min[k] <- values$TN
  OOS.FP$PL.min[k] <- values$FP
  OOS.FN$PL.min[k] <- values$FN
  
  values <- FPR_TPR( (pred1se >= val) , My[-train] )
  OOS.Ac$PL.1se[k] <- values$ACC
  OOS.TP$PL.1se[k] <- values$TP
  OOS.FP$PL.1se[k] <- values$FP
  OOS.TN$PL.1se[k] <- values$TN
  OOS.FN$PL.1se[k] <- values$FN
  
  values <- FPR_TPR( (predtheory >= val) , My[-train] )
  OOS.Ac$PL.theory[k] <- values$ACC
  OOS.TP$PL.theory[k] <- values$TP
  OOS.FP$PL.theory[k] <- values$FP
  OOS.TN$PL.theory[k] <- values$TN
  OOS.FN$PL.theory[k] <- values$FN  
  
  #Lasso
  values <- FPR_TPR( (predlassomin >= val) , My[-train] )
  OOS.Ac$L.min[k] <- values$ACC
  OOS.TP$L.min[k] <- values$TP
  OOS.TN$L.min[k] <- values$TN
  OOS.FP$L.min[k] <- values$FP
  OOS.FN$L.min[k] <- values$FN
  
  values <- FPR_TPR( (predlasso1se >= val) , My[-train] )
  OOS.Ac$L.1se[k] <- values$ACC
  OOS.TP$L.1se[k] <- values$TP
  OOS.TN$L.1se[k] <- values$TN
  OOS.FP$L.1se[k] <- values$FP
  OOS.FN$L.1se[k] <- values$FN
  
  values <- FPR_TPR( (predlassotheory >= val) , My[-train] )
  OOS.Ac$L.theory[k] <- values$ACC
  OOS.TP$L.theory[k] <- values$TP
  OOS.TN$L.theory[k] <- values$TN
  OOS.FP$L.theory[k] <- values$FP
  OOS.FN$L.theory[k] <- values$FN
  
  # Logistic Interaction
  values <- FPR_TPR( (pred.logistic.interaction >= val) , My[-train] )
  OOS.Ac$logistic.interaction[k] <- values$ACC
  OOS.TP$logistic.interaction[k] <- values$TP
  OOS.FP$logistic.interaction[k] <- values$FP
  OOS.TN$logistic.interaction[k] <- values$TN
  OOS.FN$logistic.interaction[k] <- values$FN
  
  # Logistic
  values <- FPR_TPR( (pred.logistic >= val) , My[-train] )
  OOS.Ac$logistic[k] <- values$ACC
  OOS.TP$logistic[k] <- values$TP
  OOS.TN$logistic[k] <- values$TN
  OOS.FP$logistic[k] <- values$FP
  OOS.FN$logistic[k] <- values$FN
  
  # Tree
  values <- FPR_TPR( (pred.tree >= val) , My[-train] )
  OOS.Ac$tree[k] <- values$ACC
  OOS.TP$tree[k] <- values$TP
  OOS.TN$tree[k] <- values$TN
  OOS.FP$tree[k] <- values$FP
  OOS.FN$tree[k] <- values$FN
  
  #Null
  values <- FPR_TPR( (pred.null >= val) , My[-train] )
  OOS.Ac$null[k] <- values$ACC
  OOS.TP$null[k] <- values$TP
  OOS.TN$null[k] <- values$TN
  OOS.FP$null[k] <- values$FP
  OOS.FN$null[k] <- values$FN
  
  ## We will loop this nfold times
  ## this will print the progress (iteration that finished)
  print(paste("Iteration",k,"of",nfold,"(thank you for your patience)"))
}
colMeans(OOS.R2)
m.OOS.R2 <- as.matrix(OOS.R2)
rownames(m.OOS.R2) <- c(1:nfold)
barplot(t(as.matrix(m.OOS.R2)), beside=TRUE, args.legend=c(xjust=1, yjust=0.5),
        ylab= bquote( "Out of Sample " ~ R^2), xlab="Fold", names.arg = c(1:10))


#Box plot for OOS R2 fluctuations
if (nfold >= 10){
  names(OOS.R2)[1] <-"logistic\ninteraction"
  boxplot(OOS.R2, col="plum", las = 2, ylab=expression(paste("OOS ",R^2)), xlab="", main="10-fold Cross Validation")
  names(OOS.R2)[1] <-"logistic.interaction"
}

##Zoomed box plot, without null model
if (nfold >= 10){
  names(OOS.R2)[1] <-"logistic\ninteraction"
  boxplot(OOS.R2, col="plum", las = 2, ylab=expression(paste("OOS ",R^2)), xlab="", ylim = c(0.24 ,0.28), main="10-fold Cross Validation")
  names(OOS.R2)[1] <-"logistic.interaction"
}

colMeans(OOS.Ac)

#Bar plot of accuracy
par(mar=c(1,1,1,1)) 
par(mai=c(1,1,1,1)) 
barplot(colMeans(OOS.Ac), xpd=FALSE, ylim=c(.0,1.0), xlab="Method", ylab = "Accuracy") #OOS.Ac
m.OOS.Ac <- as.matrix(OOS.Ac) #OOS.Ac
rownames(m.OOS.Ac) <- c(1:nfold)
par(mar=c(1.5,1.5,1.5,1))
par(mai=c(1.5,1.5,1.5,1))
barplot(t(as.matrix(m.OOS.Ac)), beside=TRUE, legend=TRUE, args.legend=c(x= "topright", y=0.92,bty = "n"),
        ylab= bquote( "Out of Sample Accuracy"), xlab="Fold", names.arg = c(1:10))

#Box plot for accurace
if (nfold >= 10){
  names(OOS.Ac)[1] <-"logistic\ninteraction"
  boxplot(OOS.Ac, col="plum", las = 2, ylab=expression(paste("OOS "," Accuracy")), xlab="", ylim = c(0 ,1), main="10-fold Cross Validation")
  names(OOS.Ac)[1] <-"logistic.interaction"
}

### Lets plot FPR and TPR
plot( c( 0, 1 ), c(0, 1), type="n", xlim=c(0,1), ylim=c(0,1), bty="n", xlab = "False positive rate", ylab="True positive rate")
lines(c(0,1),c(0,1), lty=2)
#
TPR = sum(OOS.TP$tree)/(sum(OOS.TP$tree)+sum(OOS.FN$tree))  
FPR = sum(OOS.FP$tree)/(sum(OOS.FP$tree)+sum(OOS.TN$tree))  
points( FPR , TPR, col = "purple" )
text( FPR + 0.06, TPR, labels=c("tree"), col = "purple")
#
TPR = sum(OOS.TP$logistic)/(sum(OOS.TP$logistic)+sum(OOS.FN$logistic))  
FPR = sum(OOS.FP$logistic)/(sum(OOS.FP$logistic)+sum(OOS.TN$logistic))  
points( FPR , TPR, col = "blue" )
text( FPR + 0.05, TPR, labels=c("LR"), col = "blue")

#
TPR = sum(OOS.TP$logistic.interaction)/(sum(OOS.TP$logistic.interaction)+sum(OOS.FN$logistic.interaction))  
FPR = sum(OOS.FP$logistic.interaction)/(sum(OOS.FP$logistic.interaction)+sum(OOS.TN$logistic.interaction))  
points( FPR , TPR, col = "red" )
text( FPR - 0.08, TPR, labels=c("LR int"), col = "red")
#
plot( c( 0, 1 ), c(0, 1), type="n", xlim=c(0,1), ylim=c(0,1), bty="n", xlab = "False positive rate", ylab="True positive rate")
lines(c(0,1),c(0,1), lty=2)

#
TPR = sum(OOS.TP$PL.min)/(sum(OOS.TP$PL.min)+sum(OOS.FN$PL.min))  
FPR = sum(OOS.FP$PL.min)/(sum(OOS.FP$PL.min)+sum(OOS.TN$PL.min))  
points( FPR , TPR, col = "purple" )
text( FPR + 0.1, TPR, labels=c("PL.min"), col = "purple")
#
TPR = sum(OOS.TP$PL.1se)/(sum(OOS.TP$PL.1se)+sum(OOS.FN$PL.1se))  
FPR = sum(OOS.FP$PL.1se)/(sum(OOS.FP$PL.1se)+sum(OOS.TN$PL.1se))  
points( FPR , TPR, col = "blue"  )
text( FPR + 0.1, TPR - 0.05, labels=c("PL.1se"), col = "blue" )
#
TPR = sum(OOS.TP$PL.theory)/(sum(OOS.TP$PL.theory)+sum(OOS.FN$PL.theory))  
FPR = sum(OOS.FP$PL.theory)/(sum(OOS.FP$PL.theory)+sum(OOS.TN$PL.theory))  
points( FPR , TPR, col = "red"  )
text( FPR - 0.12, TPR, labels=c("PL.theory"), col = "red" )
#

plot( c( 0, 1 ), c(0, 1), type="n", xlim=c(0,1), ylim=c(0,1), bty="n", xlab = "False positive rate", ylab="True positive rate")
lines(c(0,1),c(0,1), lty=2)
#

TPR = sum(OOS.TP$L.min)/(sum(OOS.TP$L.min)+sum(OOS.FN$L.min))  
FPR = sum(OOS.FP$L.min)/(sum(OOS.FP$L.min)+sum(OOS.TN$L.min))  
points( FPR , TPR, col = "purple"  )
text( FPR + 0.085, TPR, labels=c("L.min"), col = "purple" )
#
TPR = sum(OOS.TP$L.1se)/(sum(OOS.TP$L.1se)+sum(OOS.FN$L.1se))  
FPR = sum(OOS.FP$L.1se)/(sum(OOS.FP$L.1se)+sum(OOS.TN$L.1se))  
points( FPR , TPR, col = "blue" )
text( FPR + 0.085, TPR - 0.05, labels=c("L.1se"), col = "blue")
#
TPR = sum(OOS.TP$L.theory)/(sum(OOS.TP$L.theory)+sum(OOS.FN$L.theory))  
FPR = sum(OOS.FP$L.theory)/(sum(OOS.FP$L.theory)+sum(OOS.TN$L.theory))  
points( FPR , TPR, col = "red" )
text( FPR - 0.1, TPR, labels=c("L.1se"), col = "red")

#Our analysis shows that the best model to proceed with is the Lasso.Min model

#Final Model
lassomin  <- glmnet(Mx,My, family="binomial",lambda = lassoCV$lambda.min)
predlassomin <- predict(lassomin, newx=Mx, type="response")

#Note that kaggle also provided a test data set and so we have applied our model to this test data set

Mx_test<- model.matrix(Response ~ .^2, data=vehicle_insurance)[,-1] 

vehicle_insurance_test <- read.csv("test.csv")
View(vehicle_insurance_test)

vehicle_age_greater_than_2_years <- ifelse(vehicle_insurance_test$Vehicle_Age == "> 2 Years", 1, 0)
vehicle_age_between_1_and_2_years <- ifelse(vehicle_insurance_test$Vehicle_Age == "1-2 Year", 1, 0)
vehicle_age_less_than_1_year <- ifelse(vehicle_insurance_test$Vehicle_Age == "< 1 Year", 1, 0)
Gender01 <- ifelse(vehicle_insurance_test$Gender == "Male", 1, 0)
vehicle_damage <- ifelse(vehicle_insurance_test$Vehicle_Damage == "Yes", 1, 0)

vehicle_insurance <- data.frame(vehicle_insurance_test, vehicle_age_greater_than_2_years, vehicle_age_between_1_and_2_years, vehicle_age_less_than_1_year, Gender01, vehicle_damage)
vehicle_insurance <- subset(vehicle_insurance_test, select = -c(Gender, Vehicle_Age, Vehicle_Damage, id))

predlassomin_test <- predict(lassomin, newx=Mx_test, type="response")

predlassomin_test

### Cost Benefit Analysis and Deployment

cost_benefit <- read.csv("Vehicle Insurance Cost Benefit Analysis.csv")
cost_benefit

unique_age_range <- (cost_benefit$Age.Range)
unique_age_range

ggplot(data = cost_benefit, aes(x = Age.Range, y = Additional.Premium..INR., fill = Risk)) + geom_bar(stat="identity", color = "black", position= position_dodge())+theme_minimal() + xlab("Age Range") + ylab("Vehicle Insurance Premium (INR)") + ggtitle("Vehicle Insurance Premium vs Age Group") + ylim(0, 8000)

       