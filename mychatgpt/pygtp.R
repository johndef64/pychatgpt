####### Import Pychatgpt ######
#if ("reticulate" %in% installed.packages()) {
#  remove.packages("reticulate")
#}

#if (!("reticulate" %in% installed.packages())) {
#  install.packages("reticulate")
#}

library(reticulate)

# Set up a Python environment within R
#use_python("C:/Users/Utente/anaconda3/envs/torchic2/python.exe")
# Print the Python version and executable path being used by reticulate

cat("Python version:", py_config()$version, "\n") # version
cat("Python executable:", py_config()$python, "\n") # executable path

input <- function(prompt) {
  cat(prompt)  # Display the prompt to the user
  readLines(n = 1)  # Read a single line of input from the user
}

# Import the necessary Python module (pychatgpt.py)
mygpt <- import("mychatgpt")
mygpt$api_key <- mygpt$simple_decrypter(input('write your password'), mygpt$api_hash)
# Access the GPT class from the pychatgpt package
op <- mygpt$GPT()
R  <- mygpt$R
C  <- mygpt$C
R$c('ciao')

R$model <- 'gpt-4o-2024-08-06'

R$cp("correct thid code:\n")


BIO <- function(m) {
  mygpt$mendel(m, 'gpt-4o-2024-08-06')
}


#USAGE
#op$julia("ciao J")
#
#p$expand_chat("@Io ho in questo folder netOmics-case-studies/CS1_HeLa_Cell_Cycling/data/TF2DNA_datasets/pscan_hsapiens/Homo-sapiens_theoretical_TF2DNA   many many table in .pscan")
#op$expand_chat("Io volgio che tu unica un una sola tabella appendedo in vertical pe prime due colonne di questi .pscan")
#R('')