from surprise import Dataset
from surprise import KNNBasic
import pandas as pd
from surprise import Reader
from surprise.model_selection import train_test_split

# Load the full dataset
movieData = pd.read_csv('movieData.csv', index_col=0)
print(movieData.head())

# Randomly select 75% to be filled in dataframe
# This is done since the dataset is full, but Step 1 asks for 25% of empty cells.
# The fraction we want to sample is the fraction of non-null values wanted in the resulting frame 
movieData_25empty = movieData.stack().sample(frac=0.75).unstack()
print (movieData_25empty)

reader = Reader(sep=',')

data = Dataset.load_from_file(movieData_25empty, reader=reader)

# We can now use this dataset as we please, e.g. calling cross_validate
cross_validate(BaselineOnly(), data, verbose=True)

# Run "k-nearest-neighbors (k-NN) with users" algorithm
trainset = movieData_25empty.build_full_trainset()
algo = KNNBasic()
algo.fit(trainset)
