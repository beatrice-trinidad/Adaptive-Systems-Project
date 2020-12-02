from surprise import Dataset
from surprise import KNNBasic
import pandas as pd
import numpy

# Load the full dataset
# Note: movieData is cleaned from extra null columns of DatasetFull
df = pd.read_csv('movieData.csv', sep = ',',)

# Drop unnecessary columns
df.drop(df.columns[0],axis=1,inplace=True)

# Drop user name row
df.columns = df.iloc[0]

# Turn dataframe into single array
df = df.to_numpy()
numpy.concatenate(df, axis=0 )

