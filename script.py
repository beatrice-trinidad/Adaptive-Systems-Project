from surprise import Dataset
import pandas as pd

# Load the movielens-100k dataset
data = Dataset.load_builtin('ml-100k')

# Pass in column names for each data set needed
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
                    encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
                      encoding='latin-1')

# The movies file contains columns indicating the movie's genres
# Load only the first five columns of the file with usecols
m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(5),
                     encoding='latin-1')

# Create one merged DataFrame with pandas
movie_ratings = pd.merge(movies, ratings)
lens = pd.merge(movie_ratings, users)

df = pd.DataFrame(lens)

# Filter out unnecessary columns
filterData = df[['user_id','movie_id','title']]
df = filterData

# Make movies into columns
movieAsColumns = df.pivot(index='user_id', columns='movie_id', values='title')
df = movieAsColumns

print (df)

