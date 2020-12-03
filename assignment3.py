from surprise import Dataset
from surprise import KNNBasic
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
import pandas as pd
import surprise as surprise
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

##Create user array of 1-50 for every movie
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

## Make dataframe 25% empty
## Note: Not doing this anymore - Decided to split train and test set instead of changing data
#for col in df.columns:
    #df.loc[df.sample(frac=0.25).index, col] = pd.np.nan

#print(df)

## The dataframe should be converted into a dataset that Surprise can read
reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(df[['userID', 'movieID', 'ratings']], reader)

## Splitting our dataset in train and test set in a ratio of 75%:25%
trainset, testset = train_test_split(data, test_size=0.25)

#print('Number of users: ', trainset.n_users, '\n')
#print('Number of items: ', trainset.n_items, '\n')

## Run KNNBasic
#my_k = 20 # k (users/neighbors) to be adjusted
sim_options = {
    'name': 'cosine',
    'user_based': True
}
#algo = KNNBasic(k = my_k, sim_options = sim_options)

## Retrieve the trainset
#algo.fit(trainset)

## Making a prediction
uid = 6  # User ID to predict with, ex. User 8 is Harry Potter
iid = 13  # Movie ID to predict with, ex. Movie 15 is E.T.

"""So what is the prediction of Harry Potter's rating on E.T.?"""
# prediction  = algo.predict(uid, iid, r_ui = None, verbose=True)
# print(prediction)

## Predicting ratings for all pairs (u, i) that are NOT in the training set
# predictions = algo.test(testset)
#print(predictions)

# calculating mae of prediction
# mae = surprise.accuracy.mae(predictions, verbose=True)

for x in range(2, 20):
  my_k = x
  print("K value : ", x)
  algo = KNNBasic(k = my_k, sim_options = sim_options)
  algo.fit(trainset)
  prediction  = algo.predict(uid, iid, r_ui = None, verbose=True)
  predictions = algo.test(testset)
  mae = surprise.accuracy.mae(predictions, verbose=True)


