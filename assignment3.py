from surprise import Dataset
from surprise import KNNBasic
from surprise import Reader
import pandas as pd
import numpy
import numpy

## Load the full dataset
## Note: movieData is cleaned from extra null columns of DatasetFull
df = pd.read_csv('movieData.csv', sep = ',',)

## Drop unnecessary columns
df.drop(df.columns[0],axis=1,inplace=True)

## Drop user name row
df.columns = df.iloc[0]

## Turn dataframe into single array of ratings
df = df.to_numpy()
numpy.concatenate(df, axis=0 )

ratings = []

for i in range(len(df)): # Traversing through the main list
  for j in range (len(df[i])): # Traversing through each sublist
    ratings.append(df[i][j]) # Appending elements into our ratings
    
#print("Original List:",df)
#print("Flattened List:",ratings)
#print (len(ratings))

## Create user array of 1-50 for every movie
userID = []
userCount = 0

for x in range(50):
    userList = [userCount] * 20
    userID = userID + userList
    userCount = userCount + 1

#print(userID)
#print (len(userID))

## Create movie array of 1-20 for every movie
movieID = []
movieCount = 0

for x in range(20):
    movieList = [movieCount] * 50
    movieID = movieID + movieList
    movieCount = movieCount + 1

#print(movieID)
#print (len(movieID))

## Create comprehensive dataframe of all data
ratings_dict = {'userID': userID,
                'movieID': movieID,
                'ratings': ratings}

df = pd.DataFrame(ratings_dict)

#print(df)

## The dataframe should be converted into a dataset that Surprise can read
reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(df[['userID', 'movieID', 'ratings']], reader)
