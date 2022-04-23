#############################################################
# Machine Learning for Social Sciences
# Felix Hagemeister
# 2022

# R Script: Regularisation with Lasso, using glmnet()

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
library(tidyverse)    # for data reading wrangling and visualization
library(AmesHousing)  # data on housing and prices
library(rsample)      # for train/test split
library(glmnet)       # for lasso regression


# load housing data
# --------------------------------------------------
df <- make_ames() 

# Goal: predict housing price
summary(df$Sale_Price)

# Split data into train and test sample
# --------------------------------------------------
set.seed(20220318)
split <- initial_split(df,prop =0.70)
df_train <- training(split)
df_test <- testing(split)

# Pre-processing
# --------------------------------------------------
# create X and y training data, leave out constant
X_train <- model.matrix(Sale_Price ~ ., data=df_train)[,-1]
y_train <- df_train$Sale_Price

dim(X_train) # note that factors are expanded into dummy variables

# Lasso regression
# ----------------------------------------------

?cv.glmnet
fit_lasso <- cv.glmnet(x = X_train, y = y_train)
plot(fit_lasso)

# extract minimum MSE of lasso path
fit_lasso$lambda.min

# get coefficients
coefs_lasso <- enframe(coef(fit_lasso, s = "lambda.min")[, 1])


# Prediction using test sample
# -------------------------------------------

?predict.cv.glmnet

X_test <- model.matrix(Sale_Price ~ . , data=df_test)[,-1]

# for each row in X_test, predict using fitted beta coefficients from training, then plot

df_test %>%
  mutate(
    pred_lasso = as.vector(predict(fit_lasso,
                                   newx = X_test,
                                   s="lambda.min")) 

  )%>%
  ggplot(aes(x = Sale_Price, y = pred_lasso)) +
  geom_point(alpha= 0.5) +
  coord_equal()


### END