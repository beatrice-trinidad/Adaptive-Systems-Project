from collections import defaultdict
from surprise import Dataset
from surprise import SVD, KNNBasic
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

######################################################
#### TASK #1
######################################################

## Splitting our dataset in train and test set in a ratio of 75%:25%
trainset, testset = train_test_split(data, test_size=0.25)

#print('Number of users: ', trainset.n_users, '\n')
#print('Number of items: ', trainset.n_items, '\n')

## Define algorithm options
sim_options = {
    'name': 'pearson',
    'user_based': True
}

## Making a prediction
uid = 6  # User ID to predict with, ex. User 8 is Harry Potter
iid = 13  # Movie ID to predict with, ex. Movie 15 is E.T.

res_array = []
min_MAE = 0

for x in range(1, 50):
  my_k = x
  print("K value : ", x)
  ## Run KNNBasic
  algo = KNNBasic(k = my_k, sim_options = sim_options)
  ## Retrieve the trainset
  algo.fit(trainset)
  ## Predicting ratings for all pairs (u, i) that are NOT in the training set
  predictions = algo.test(testset) 
  ## Calculate MAE and append it to result array
  mae = surprise.accuracy.mae(predictions, verbose=True)
  res_array.append(mae)

print("***** MAEs for ALL USERS with 25 PERCENT EMPTY MATRIX ******")
min_MAE = min(res_array)
print(res_array)
print("***** RESULTS WITH KNN with 25 PERCENT EMPTY MATRIX ******")
print("Minimum Mean Absolute Error: ", min_MAE)
print("Best K: ", res_array.index(min_MAE)+1)

def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)

top_n = get_top_n(predictions, n=10)

# Print the 10 recommended items for each user
print("***** TOP 10 RECOMMENDED ITEMS FOR EACH USER WITH KNN ******")
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])

top_n = get_top_n(predictions, n=5)

# Print the 5 recommended items for each user
print("***** TOP 5 RECOMMENDED ITEMS FOR EACH USER WITH KNN ******")
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])

top_n = get_top_n(predictions, n=2)

# Print the 2 recommended items for each user
print("***** TOP 2 RECOMMENDED ITEMS FOR EACH USER WITH KNN ******")
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])


######################################################
#### SPARSITY PROBLEM: TASK #2
######################################################

## Splitting our dataset in train and test set in a ratio of 25%:75%
trainset, testset = train_test_split(data, test_size=0.75)

#print('Number of users: ', trainset.n_users, '\n')
#print('Number of items: ', trainset.n_items, '\n')

## Define algorithm options
sim_options = {
    'name': 'pearson',
    'user_based': True
}

## Making a prediction
uid = 6  # User ID to predict with, ex. User 8 is Harry Potter
iid = 13  # Movie ID to predict with, ex. Movie 15 is E.T.

res_array = []
min_MAE = 0

for x in range(1, 50):
  my_k = x
  print("K value : ", x)
  ## Run KNNBasic
  algo = KNNBasic(k = my_k, sim_options = sim_options)
  ## Retrieve the trainset
  algo.fit(trainset)
  ## Predicting ratings for all pairs (u, i) that are NOT in the training set
  predictions = algo.test(testset) 
  ## Calculate MAE and append it to result array
  mae = surprise.accuracy.mae(predictions, verbose=True)
  res_array.append(mae)

print("***** MAEs for ALL USERS with 75 PERCENT EMPTY MATRIX ******")
min_MAE = min(res_array)
print(res_array)
print("***** RESULTS WITH KNN with 75 PERCENT EMPTY MATRIX ******")
print("Minimum Mean Absolute Error: ", min_MAE)
print("Best K: ", res_array.index(min_MAE)+1)

def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)

top_n = get_top_n(predictions, n=10)

# Print the 10 recommended items for each user
print("***** TOP 10 RECOMMENDED ITEMS FOR EACH USER WITH KNN ******")
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])

top_n = get_top_n(predictions, n=5)

# Print the 5 recommended items for each user
print("***** TOP 5 RECOMMENDED ITEMS FOR EACH USER WITH KNN ******")
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])

top_n = get_top_n(predictions, n=2)

# Print the 2 recommended items for each user
print("***** TOP 2 RECOMMENDED ITEMS FOR EACH USER WITH KNN ******")
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])

######################################################
#### MITIGATION OF SPARSITY PROBLEM: TASK #3
######################################################

## Splitting our dataset in train and test set in a ratio of 25%:75%
trainset, testset = train_test_split(data, test_size=0.75)

def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# First train an SVD algorithm on the movielens dataset
algo = SVD()
algo.fit(trainset)

# Than predict ratings for all pairs (u, i) that are NOT in the training set
testset = trainset.build_anti_testset()
predictions = algo.test(testset)

top_n = get_top_n(predictions, n=10)

# Print the recommended items for each user
print("***** TOP 10 RECOMMENDED ITEMS FOR EACH USER WITH SVD ******")
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])


## Calculate SVD's MAE 
print("***** MAE for SVD ******")
surprise.accuracy.mae(predictions, verbose=True)
