#############################################################
# Machine Learning for Social Sciences
# Felix Hagemeister
# 2022

# R Script 0: Set-Up

# clear environment
# -------------------------------
rm(list = ls())

# set custom path using system user name (need to enter your path)
# -------------------------------
if (Sys.getenv("USERNAME") == "felix"){
  setwd("C:/Users/felix/Dropbox/HfP/Teaching/WiSe21/ML/")}

if (Sys.getenv("USERNAME") == "[YOUR USER NAME HERE]"){
  setwd("[YOUR PATH HERE")}


# install packages if needed (uncomment to run)
# -----------------------------------
#install.packages("gamlr")   # for Gamma-Lasso Regression

# load packages
# -----------------------------------
library(gamlr)

# see documentation
# -----------------------------------
?gamlr()

# A primer for what is to come
# -----------------------------------

# simulate some data
set.seed(20220422)
n <- 1000
x1 <- rnorm(n, mean = 1, sd = 1)
x2 <- rnorm(n, mean = 2, sd = 0.5)
x3 <- rnorm(n, mean = 3, sd = 2)
y <- 4 + 3*x1 + (-1)*x2 +  rnorm(n)
X <- cbind(x1,x2,x3)

# descriptive statistics
summary(y)
summary(X)

# simple plot
plot(y ~ x1, col= "blue")
abline(lm(y ~ x1), col= "red")

# simple regression, using lm()
m1 <- lm(y ~ x1 + x2 + x3)
summary(m1)

# simple regression, using glm()
m2 <- glm(y ~ x1 + x2 + x3, family="gaussian")
summary(m2)

# fit lasso regression
fit <- gamlr(X, y)

# Plot Lasso Path
# Interpretation: read from right to left, 
# x-axis is lambda, the increasing penalty for more covariates
# y-axis is coefficient size
# number nonzero coefficients on top
plot(cvfit,  col="navy") # but: this alone cannot do model selection

# fit lasso regressions with OOS cross-validation
?cv.gamlr()
cvfit <- cv.gamlr(X, y)

# plot OOS errors for different models
plot(cvfit)

# extract minimum MSE of lasso path
cvfit$lambda.min
coefs_lasso <- enframe(coef(cvfit, s = "min")[, 1])

# get coefficients for best-performing model
coefs_lasso


### END