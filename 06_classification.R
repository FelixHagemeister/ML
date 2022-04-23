#############################################################
# Machine Learning for Social Sciences
# Felix Hagemeister
# 2022

# R Script: Binary classification with logistic lasso, using gamlr()


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
# -------------------------------
library(gamlr)        # for lasso regression

# load German Credit data
# --------------------------------------------------

credit <- read.csv("data/credit.csv")

# Pre-process: generate variables for prediction
# --------------------------------------------------

## re-level the credit history and checking account status
table(credit$history)
credit$history = factor(credit$history, levels=c("A30","A31","A32","A33","A34"))
levels(credit$history) = c("good","good","poor","poor","terrible")
## a few others
credit$foreign <- factor(credit$foreign, levels=c("A201","A202"), labels=c("foreign","german"))
credit$rent <- factor(credit$housing=="A151")
credit$purpose <- factor(credit$purpose, levels=c("A40","A41","A42","A43","A44","A45","A46","A47","A48","A49","A410"))
levels(credit$purpose) <- c("newcar","usedcar",rep("goods/repair",4),"edu",NA,"edu","biz","biz")

credit <- credit[,c("Default", "duration", "amount",
                    "installment", "age", "history",
                    "purpose", "foreign", "rent")]

# Define X matrix
# --------------------------------------------------
# use all pairwise interactions of inputs, 
# and all factor levels without reference categories.

source("code/naref.R")
credx <- sparse.model.matrix( Default ~ .^2, data=naref(credit))[,-1]

# Define response variable y
# --------------------------------------------------

default <- credit$Default

# Run Gamma Lasso Regression with k-fold cross validation
# --------------------------------------------------

credscore <- cv.gamlr(credx, default, family="binomial", verb=TRUE)


# Show regularisation path and CV results
# --------------------------------------------------

par(mfrow=c(1,2))
plot(credscore$gamlr)
plot(credscore)

# Get number of non-zero coefficients in the different models
# --------------------------------------------------

sum(coef(credscore, s="min")!=0) # min
sum(coef(credscore$gamlr)!=0) # AICc
sum(coef(credscore)!=0) # 1se
sum(coef(credscore$gamlr, s=which.min(AIC(credscore$gamlr)))!=0) # AIC
sum(coef(credscore$gamlr, s=which.min(BIC(credscore$gamlr)))!=0) # BIC

# Calculate the OOS R2
# --------------------------------------------------
# the OOS R^2 is (1- RSS/TSS)
# note that cvm	object is "Mean OOS deviance by gamlr\$lambda"

1 - credscore$cvm[credscore$seg.min]/credscore$cvm[1]

# Show underlying (in-sample) probabilites for outcome
# --------------------------------------------------

pred <- predict(credscore$gamlr, credx, type="response")
pred <- drop(pred) # remove the sparse Matrix formatting
boxplot(pred ~ default, xlab="default", ylab="prob of default", col=c("pink","dodgerblue"))

# Calculate misclassification rates
# --------------------------------------------------
rule <- 1/5 # move this around to see how these change

sum( (pred>rule)[default==0] )/sum(pred>rule) ## false positive rate
sum( (pred<rule)[default==1] )/sum(pred<rule) ## false negative rate

sum( (pred>rule)[default==1] )/sum(default==1) ## sensitivity
sum( (pred<rule)[default==0] )/sum(default==0) ## specificity

# OOS ROC curve
# --------------------------------------------------
# refit the model using only 1/2 of data
test <- sample.int(1000,500)
credhalf <- gamlr(credx[-test,], default[-test], family="binomial")
predoos <- predict(credhalf, credx[test,], type="response")
defaultoos <- default[test]

## roc curve and fitted distributions
## plot the ROC curve for classification of y with p
roc <- function(p,y, ...){
  y <- factor(y)
  n <- length(p)
  p <- as.vector(p)
  Q <- p > matrix(rep(seq(0,1,length=100),n),ncol=100,byrow=TRUE)
  specificity <- colMeans(!Q[y==levels(y)[1],])
  sensitivity <- colMeans(Q[y==levels(y)[2],])
  plot(1-specificity, sensitivity, type="l", ...)
  abline(a=0,b=1,lty=2,col=8)
}


par(mai=c(.9,.9,.2,.1), mfrow=c(1,2))
roc(p=pred, y=default, bty="n", main="in-sample")
## our 1/5 rule cutoff
points(x= 1-mean((pred<.2)[default==0]), 
       y=mean((pred>.2)[default==1]), 
       cex=1.5, pch=20, col='red') 
## a standard `max prob' (p=.5) rule
points(x= 1-mean((pred<.5)[default==0]), 
       y=mean((pred>.5)[default==1]), 
       cex=1.5, pch=20, col='blue') 
legend("bottomright",fill=c("red","blue"),
       legend=c("p=1/5","p=1/2"),bty="n",title="cutoff")

r2 <- roc(p=predoos, y=defaultoos, bty="n", main="out-of-sample")
## our 1/5 rule cutoff
points(x= 1-mean((predoos<.2)[defaultoos==0]), 
       y=mean((predoos>.2)[defaultoos==1]), 
       cex=1.5, pch=20, col='red') 
## a standard `max prob' (p=.5) rule
points(x= 1-mean((predoos<.5)[defaultoos==0]), 
       y=mean((predoos>.5)[defaultoos==1]), 
       cex=1.5, pch=20, col='blue') 

## plot a mosaic
# --------------------------------------------------

par(mai=c(.8,.8,.1,.1))
plot(factor(Default) ~ history, data=credit, col=c(8,2), ylab="Default") ## surprise!
## the dangers of choice-based sampling!  


### END