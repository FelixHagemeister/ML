#############################################################
# Machine Learning for Social Sciences
# Felix Hagemeister
# 2022

# R Script: Regression and Repeated K-Fold Out-Of-Sample (OOS) Validation, using caret

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
# -----------------------------------
library(tidyverse)   # for data reading wrangling and visualization
library(caret)       # standard ML package (but now replaced by tidymodels)
library(datarium)    # for data

# load data
# -----------------------------------

# loading the dataset
data("marketing", package = "datarium")

# inspecting the dataset
head(marketing)


# repeated K-fold cross-validation
# -----------------------------------

# setting seed to generate a
# reproducible random sampling
set.seed(125)

# defining training control as
# repeated cross-validation and
# value of K is 10 and repetition is 3 times
train_control <- trainControl(method = "repeatedcv",
                              number = 10, repeats = 3)

# training the model by assigning sales column
# as target variable and rest other column
# as independent variable
model <- train(sales ~., data = marketing,
               method = "lm",
               trControl = train_control)

# printing model performance metrics
# along with other details
print(model)

### END