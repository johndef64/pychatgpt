####### Import Pychatgpt ######
#if ("reticulate" %in% installed.packages()) {
#  remove.packages("reticulate")
#}

#if (!("reticulate" %in% installed.packages())) {
#  install.packages("reticulate")
#}

library(reticulate)

# Set up a Python environment within R
use_python("C:/Users/Utente/anaconda3/envs/torchic/python.exe") # Change with you Python env

# Import the necessary Python module (pychatgpt.py)
op <- import("pychatgpt")

R <- function(m) {
  op$roger(m, 'gpt-4o-2024-08-06')
}

Rp <- function(m) {
  m <- paste0(m, op$pc$paste())
  R(m)
}


BIO <- function(m) {
  op$mendel(m, 'gpt-4o-2024-08-06')
}

BIOp <- function(m) {
  m <- paste0(m, op$pc$paste())
  BIO(m)
}

Py <- function(m) {
  op$delamain(m, 'gpt-4o-2024-08-06')
}

Pyp <- function(m) {
  m <- paste0(m, op$pc$paste())
  Py(m)
}
