#############################################################
# Machine Learning for Social Sciences
# Felix Hagemeister
# 2022

# R Script: Regression and K-Fold Out-Of-Sample (OOS) Validation by hand

# clear environment
# -------------------------------
rm(list = ls())

# set custom path using system user name
# -------------------------------
if (Sys.getenv("USERNAME") == "felix"){
  setwd("C:/Users/felix/Dropbox/HfP/Teaching/WiSe21/ML/")}
if (Sys.getenv("USERNAME") == "[YOUR USER NAME HERE]"){
  setwd("[YOUR PATH HERE")}


# load packages
# --------------------------------------------------
library(tidyverse)   # for data reading wrangling and visualization
library(ggplot2)     # for plotting


# load titanic data
# --------------------------------------------------

df <- read_csv("data/titanic_sample.csv")  

# explore variables
# --------------------------------------------------

str(df)
summary(df$Survived)
table(df$Survived, df$Sex)

# predict survival in sample (IS)
# --------------------------------------------------

# Simple Linear Regression
m1 <- lm(Survived ~ Pclass + Sex + Age , data=df)
summary(m1)

# get predicted values
df$m1_pred <- predict(m1)

# visualize
plot(Survived ~ m1_pred, data=df)
abline(lm(Survived ~ m1_pred, data = df), col = "blue")

# get IN-SAMPLE R2
summary(m1)$r.squared

# Logistic regression
m2 <- glm(Survived ~ Pclass + Sex + Age , data=df,
          family="binomial")
summary(m2)

# get predicted values
df$m2_pred <- predict(m2)

# visualize
ggplot(df, aes(x=m2_pred, y=Survived)) + 
  geom_point(alpha=.5) +
  stat_smooth(method="glm", se=FALSE, method.args = list(family=binomial),
              col="red", lty=2)

# define deviance and R2 for linear and logistic regression models
# --------------------------------------------------------------------

## pred must be probabilities (0<pred<1) for binomial
deviance <- function(y, pred, family=c("gaussian","binomial")){
  family <- match.arg(family)
  if(family=="gaussian"){
    return( sum( (y-pred)^2 ) )
  }else{
    if(is.factor(y)) y <- as.numeric(y)>1
    return( -2*sum( y*log(pred) + (1-y)*log(1-pred) ) )
  }
}

## get null deviance too, and return R2
R2 <- function(y, pred, family=c("gaussian","binomial")){
  fam <- match.arg(family)
  if(fam=="binomial"){
    if(is.factor(y)){ y <- as.numeric(y)>1 }
  }
  dev <- deviance(y, pred, family=fam)
  dev0 <- deviance(y, mean(y), family=fam)
  return(1-dev/dev0)
}


# get IN-SAMPLE R2
1 - m2$deviance/ m2$null.deviance

# predict survival out of sample (OOS) with k-fold validation
# --------------------------------------------------

n= nrow(df) # number of observations
K= 10       # number of 'folds'
# create a vector of fold memberships (random order)
foldid <- rep(1:K, each=ceiling(n/K))[sample(1:n)]
# create an empty dataframe of results
OOS <- data.frame(m1=rep(NA,K), m2=rep(NA,K))

# run OOS experiment as a for loop 'by hand':

for(k in 1:K){
  
  train <- which(foldid!=k) # train on all but fold 'k'
  
  # fit linear and logistic regression in training sample (leaving out k-th fold)
  m1 <- glm(Survived ~ Pclass + Sex + Age,
           data= df, subset = train, family= "gaussian")
  m2 <- glm(Survived ~ Pclass + Sex + Age,
            data= df, subset = train, family = "binomial")
  
  # get OOS predictions in k-th fold
  m1_pred <- predict(m1, newdata=df[-train,], type="response")
  m2_pred <- predict(m2, newdata=df[-train,], type="response")
  
  # calculate and log R2 in k-th fold
  OOS$m1[k] <- R2(y=df$Survived[-train], pred=m1_pred , family="gaussian")
  OOS$m2[k] <- R2(y=df$Survived[-train], pred=m2_pred , family="gaussian")
  
## print progress
cat(k, " ")
}

# summary
summary(OOS)

# Task: try to increase OOS R2 by using different model specifications
# Some suggestions: add or remove some variables, try interactions, use dummies, standardize variables

### END
