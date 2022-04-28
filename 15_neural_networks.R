#############################################################
# Machine Learning for Social Sciences
# Felix Hagemeister
# 2022

# R Script: Deep Neural Networks (DNNs)

# Clear environment
# -------------------------------
rm(list = ls())

# Set custom path using system user name
# -------------------------------
if (Sys.getenv("USERNAME") == "felix"){
  setwd("C:/Users/felix/Dropbox/HfP/Teaching/WiSe21/ML/")}
if (Sys.getenv("USERNAME") == "[YOUR USER NAME HERE]"){
  setwd("[YOUR PATH HERE")}


# Load packages
# -------------------------------
# Helper packages
library(dplyr)         # for basic data wrangling

# Modeling packages
reticulate::install_miniconda()
reticulate::conda_create(
  envname = "digitaldlsorter-env", 
  packages = "python==3.7.11"
)
tensorflow::install_tensorflow(
  method = "conda", 
  conda = reticulate::conda_binary("auto"), 
  envname = "digitaldlsorter-env", 
  version = "2.1.0-cpu",
)

library(keras)         # for fitting DNNs
tensorflow::use_condaenv("digitaldlsorter-env")

library(tfruns)        # for additional grid search & model training functions

# Modeling helper package - not necessary for reproducibility
library(tfestimators)  # provides grid search & model training interface



# Load MNIST training data
# -------------------------------

mnist <- dslabs::read_mnist()
mnist_x <- mnist$train$images
mnist_y <- mnist$train$labels

# Rename columns and standardize feature values
colnames(mnist_x) <- paste0("V", 1:ncol(mnist_x))
mnist_x <- mnist_x / 255

# One-hot encode response
mnist_y <- to_categorical(mnist_y, 10)


# Implement DNN
# --------------------------------------

# Initiate our sequential feedforward DNN architecture, 
# add layers, 
# specify activation functions
# incorporate backpropagation


model <- keras_model_sequential() %>%
  
  # Network architecture with activation functions
  layer_dense(units = 128, activation = "relu", input_shape = ncol(mnist_x)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax") %>%
  
  # Backpropagation
  compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
  )


# Train the model
# --------------------------

# tuning parameters:
# batch_size: Batch sizes to run through the mini-batch SDG c
# epochs: An epoch describes the number of times the algorithm sees the entire data set.
# validation_split: The model will hold out XX% of the data for OOS performance computation

fit1 <- model %>%
  fit(
    x = mnist_x,
    y = mnist_y,
    epochs = 25,
    batch_size = 128,
    validation_split = 0.2,
    verbose = FALSE
  )


# Display output
fit1
plot(fit1)

# Tuning 1: Capacity and Batch normalisation
# --------------------------
# Capacity: add layers
# Batch normalisation: add layer_batch_normalization() after each layer

model_w_norm <- keras_model_sequential() %>%
  
  # Network architecture with batch normalization
  layer_dense(units = 256, activation = "relu", input_shape = ncol(mnist_x)) %>%
  layer_batch_normalization() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dense(units = 10, activation = "softmax") %>%
  
  # Backpropagation
  compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_rmsprop(),
    metrics = c("accuracy")
  )


fit2 <- model_w_norm %>%
  fit(
    x = mnist_x,
    y = mnist_y,
    epochs = 25,
    batch_size = 128,
    validation_split = 0.2,
    verbose = FALSE
  )

# Display output
fit2
plot(fit2)

# Tuning 2: Regularisation with dropout
# --------------------------
model_w_drop <- keras_model_sequential() %>%
  
  # Network architecture with 20% dropout
  layer_dense(units = 256, activation = "relu", input_shape = ncol(mnist_x)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 10, activation = "softmax") %>%
  
  # Backpropagation
  compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_rmsprop(),
    metrics = c("accuracy")
  )

# Tuning 3: Learning rate optimisation
# --------------------------

model_w_adj_lrn <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = ncol(mnist_x)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 10, activation = "softmax") %>%
  compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_adam(),
    metrics = c('accuracy')
  ) %>%
  fit(
    x = mnist_x,
    y = mnist_y,
    epochs = 35,
    batch_size = 128,
    validation_split = 0.2,
    callbacks = list(
      callback_early_stopping(patience = 5),
      callback_reduce_lr_on_plateau(factor = 0.05)
    ),
    verbose = FALSE
  )

model_w_adj_lrn

# Optimal
min(model_w_adj_lrn$metrics$val_loss)
max(model_w_adj_lrn$metrics$val_acc)

# plot
plot(model_w_adj_lrn)


# Grid Search for Optimal Tuning Parameters
# --------------------------

# first establish flags for the different hyperparameters of interest.
FLAGS <- flags(
  # Nodes
  flag_numeric("nodes1", 256),
  flag_numeric("nodes2", 128),
  flag_numeric("nodes3", 64),
  # Dropout
  flag_numeric("dropout1", 0.4),
  flag_numeric("dropout2", 0.3),
  flag_numeric("dropout3", 0.2),
  # Learning paramaters
  flag_string("optimizer", "rmsprop"),
  flag_numeric("lr_annealing", 0.1)
)

# Next, we incorprate the flag parameters within our model:
model <- keras_model_sequential() %>%
  layer_dense(units = FLAGS$nodes1, activation = "relu", input_shape = ncol(mnist_x)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = FLAGS$dropout1) %>%
  layer_dense(units = FLAGS$nodes2, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = FLAGS$dropout2) %>%
  layer_dense(units = FLAGS$nodes3, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = FLAGS$dropout3) %>%
  layer_dense(units = 10, activation = "softmax") %>%
  compile(
    loss = 'categorical_crossentropy',
    metrics = c('accuracy'),
    optimizer = FLAGS$optimizer
  ) %>%
  fit(
    x = mnist_x,
    y = mnist_y,
    epochs = 35,
    batch_size = 128,
    validation_split = 0.2,
    callbacks = list(
      callback_early_stopping(patience = 5),
      callback_reduce_lr_on_plateau(factor = FLAGS$lr_annealing)
    ),
    verbose = FALSE
  )

# execute the grid search, using 5% of total models (145 of the 2916 in this example)
# !this grid seach will take about 1-2 hours to run

# Run various combinations of dropout1 and dropout2
runs <- tuning_run("scripts/mnist-grid-search.R", 
                   flags = list(
                     nodes1 = c(64, 128, 256),
                     nodes2 = c(64, 128, 256),
                     nodes3 = c(64, 128, 256),
                     dropout1 = c(0.2, 0.3, 0.4),
                     dropout2 = c(0.2, 0.3, 0.4),
                     dropout3 = c(0.2, 0.3, 0.4),
                     optimizer = c("rmsprop", "adam"),
                     lr_annealing = c(0.1, 0.05)
                   ),
                   sample = 0.05
)

runs %>% 
  filter(metric_val_loss == min(metric_val_loss)) %>% 
  glimpse()

## END

