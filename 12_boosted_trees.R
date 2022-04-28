#############################################################
# Machine Learning for Social Sciences
# Felix Hagemeister
# 2022

# R Script: Boosted Trees


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
# ------------------------------
library(recipes)  # for matrix encoding
library(rsample)  # for train/test split
library(xgboost)  # for fitting extreme gradient boosting
library(vip)      # for variable importance plot


# load Ames housing data
# -------------------------------
ames <- AmesHousing::make_ames()


# split data into training and testing
# -------------------------------

set.seed(123)
split <- initial_split(ames, prop = 0.7, 
                       strata = "Sale_Price")
ames_train  <- training(split)
ames_test   <- testing(split)

# additional data preparation
# -------------------------------

# xgboost requires a matrix input for the features and the response to be a vector.
# To provide a matrix input of the features,
# we need to encode our categorical variables numerically
# (i.e. one-hot encoding, label encoding).
# The following numerically label encodes all categorical features 
# and converts the training data frame to a matrix.

xgb_prep <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_integer(all_nominal()) %>%
  prep(training = ames_train, retain = TRUE) %>%
  juice()

X <- as.matrix(xgb_prep[setdiff(names(xgb_prep), "Sale_Price")])
Y <- xgb_prep$Sale_Price


# run xgboosted 
# -------------------------------

?xgb.cv()

set.seed(20220427)
ames_xgb <- xgb.cv(
  data = X,
  label = Y,
  nrounds = 6000,
  objective = "reg:squarederror",
  early_stopping_rounds = 50, 
  nfold = 10,
  params = list(
    eta = 0.1,
    max_depth = 3,
    min_child_weight = 3,
    subsample = 0.8,
    colsample_bytree = 1.0),
  verbose = 0
)  

# minimum test CV RMSE
min(ames_xgb$evaluation_log$test_rmse_mean)
## [1] 20488


# Find optimal hyperparameters (might use grid search)
# --------------------------------------

# optimal parameter list
params <- list(
  eta = 0.01,
  max_depth = 3,
  min_child_weight = 3,
  subsample = 0.5,
  colsample_bytree = 0.5
)

# Fit final model
# --------------------------------------

# train final model
xgb.fit.final <- xgboost(
  params = params,
  data = X,
  label = Y,
  nrounds = 3944,
  objective = "reg:squarederror",
  verbose = 0
)

# Feature importance (using impurity/gain)
# --------------------------------------

# variable importance plot
vip::vip(xgb.fit.final) 

## END