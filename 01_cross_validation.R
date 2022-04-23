#############################################################
# Machine Learning for Social Sciences
# Felix Hagemeister
# 2022

# R Script: Types of Cross-Validation


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
library(tidyverse) # meta-package for data science
library(caret)     # for machine learning (now replaced by tidymodels package)
library(datarium)  # for data


# load data
# -------------------------------

data("marketing", package = "datarium")

# inspecting the dataset
head(marketing)

# Method 1: validation set approach
# -------------------------------

# setting seed to generate a
# reproducible random sampling
set.seed(20220423)

# creating training data as 80% of the dataset
random_sample <- createDataPartition(marketing$sales, p = 0.8, list = FALSE)

# generating training dataset
# from the random_sample
training_dataset  <- marketing[random_sample, ]

# generating testing dataset
# from rows which are not
# included in random_sample
testing_dataset <- marketing[-random_sample, ]

# Building the model

# training the model by assigning sales column
# as target variable and rest other columns
# as independent variables
model <- lm(sales ~., data = training_dataset)

# predicting the target variable
predictions <- predict(model, testing_dataset)

# computing model performance metrics
data.frame( R2 = R2(predictions, testing_dataset$sales),
            RMSE = RMSE(predictions, testing_dataset$sales),
            MAE = MAE(predictions, testing_dataset$sales))


# Method 2: Leave one out cross validation ("LOOCV")
# ---------------------------------------

# defining training control
# as Leave One Out Cross Validation
train_control <- trainControl(method = "LOOCV")

# training the model by assigning sales column
# as target variable and rest other column
# as independent variable
model <- train(sales ~., data = marketing,
               method = "lm",
               trControl = train_control)

# printing model performance metrics
# along with other details
print(model)


# Method 3: K-fold cross-validation
# ---------------------------------------

# setting seed to generate a
# reproducible random sampling
set.seed(125)

# defining training control
# as cross-validation and
# value of K equal to 10
train_control <- trainControl(method = "cv",
                              number = 10)

# training the model by assigning sales column
# as target variable and rest other column
# as independent variable
model <- train(sales ~., data = marketing,
               method = "lm",
               trControl = train_control)

# printing model performance metrics
# along with other details
print(model)


# Method 4: Repeated K-fold cross-validation
# ---------------------------------------

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


## END
