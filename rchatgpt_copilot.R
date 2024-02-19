library(reticulate)

# Set up a Python environment within R
use_python("C:/Users/Utente/anaconda3/envs/myenv/python.exe")

# Import the necessary Python module (pychatgpt.py)
op <- import("pychatgpt")

# Call a Python function
op$bestie("Hi there! Wath's up bro")
print(op$reply)

# Call R Copilot
op$clearchat()
m <- "Make a numeric matrix example and analyze it with tidyverse package"
op$roger(m)
print(op$reply)
# --> Paste below the generated code with Ctrl+V

###############################
# Load the tidyverse package
library(tidyverse)

# Create a numeric matrix
my_matrix <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3)

# Print the matrix
print(my_matrix)

# Analyze the matrix using tidyverse functions
my_matrix %>%
  as.data.frame() %>%
  gather(key = "column", value = "value") %>%
  group_by(column) %>%
  summarise(
    mean_value = mean(value),
    max_value = max(value),
    min_value = min(value)
  )
###############################


m <- "Now add more random data and plot them using tidyverse features"
op$roger(m)
print(op$chat_gpt)

###############################
# Load the required packages
library(tidyverse)

# Generate example data
set.seed(123) # Set the seed for reproducibility
x <- rnorm(100, mean = 0, sd = 1) # Generate 100 random numbers from a normal distribution
y <- 2 * x + rnorm(100, mean = 0, sd = 0.5) # Create y as a linear function of x with some random noise

# Create a tibble from the generated data
data <- tibble(x = x, y = y)

# Plot the data using ggplot2
ggplot(data, aes(x = x, y = y)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "X-axis", y = "Y-axis", title = "Scatter plot of X and Y")
###############################
