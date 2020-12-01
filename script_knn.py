from surprise import Dataset
import pandas as pd

# Load the full dataset
movieData = pd.read_csv('movieData.csv', index_col=0, )
print(movieData.head())

# Randomly select 75% of rows in dataframe
# This is done since the dataset is full, but Step 1 asks for 25% of empty cells.
movieData_one = movieData.sample(frac=0.75)
print(movieData_one)

# Pass in column names for each data set needed


