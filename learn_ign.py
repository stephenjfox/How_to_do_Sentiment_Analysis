import pandas as pd

# Easy is to map a review to simply "positive" or "negative"
# Hard maps the reviews to their full descriptions:
# learned more about the data set: "Masterpiece", "Unbearable", "Disaster", "Okay" are also rating phrases
# Masterpiece, Amazing, Great, Good, Okay, Mediocre, Awful, Painful, Unbearable, Disaster

difficulty = { 'easy': { 'output_nodes': 2 }, 'hard': { 'output_nodes': 10 } }

# Data Preparation
## 0. Read data
## 1. Clean data (i.e. remove redundancies)
df = pd.read_csv('cleaned_ign_reviews.csv')


## 2. Transform
### * -> Numeric

print(df[:10])
### Normalize


## 3. Reduction
### Dimensionality Reduction
### Feature Extraction


# Load into TFLearn model: http://tflearn.org/data_utils/#load_csv
# TODO: After getting the data into numeric form I should do the following

## Taken from the notebook on "your first neural network"
## translate each category into multiple columns that are binary.
## Then it will be all the easier for the network to work with it.

# dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
# for each in dummy_fields:
#     dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
#     rides = pd.concat([rides, dummies], axis=1)
#
# fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
#                   'weekday', 'atemp', 'mnth', 'workingday', 'hr']
# data = rides.drop(fields_to_drop, axis=1)
# data.head()
