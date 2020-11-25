from surprise import Dataset

# Load the movielens-100k dataset (download it if needed),
data = Dataset.load_builtin('ml-100k')

print("Data successfully loaded")