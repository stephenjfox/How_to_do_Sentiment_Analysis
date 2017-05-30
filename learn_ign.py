import pandas as pd

# Easy is to map a review to simply "positive" or "negative"
# Hard maps the reviews to their full descriptions:
# learned more about the data set: "Masterpiece", "Unbearable", "Disaster", "Okay" are also rating phrases
# Masterpiece, Amazing, Great, Good, Okay, Mediocre, Bad, Awful, Painful, Unbearable, Disaster

phrase_to_numeric = {
    'Masterpiece': 0,
    'Amazing': 1,
    'Great': 2,
    'Good': 3,
    'Okay': 4,
    'Mediocre': 5,
    'Bad': 6,
    'Awful': 7,
    'Painful': 8,
    'Unbearable': 9,
    'Disaster': 10
}

# Data Preparation
## 0. Read data
## 1. Clean data (i.e. remove redundancies)
reviews = pd.read_csv('cleaned_ign_reviews.csv')


## 2. Transform
### * -> Numeric

# convert 'Y'/'N' -> 1/0
reviews['editors_choice'] = reviews['editors_choice'].apply(lambda l: 1 if l == 'Y' else 0)

# phrases to numbers
reviews['score_phrase'] = reviews['score_phrase'].apply(lambda phrase: phrase_to_numeric[phrase])

unique_genres = reviews.drop_duplicates(['genre'])

genre_to_numeric = { genre: i for i, genre in enumerate(unique_genres['genre'].sort_values())}

reviews['genre'] = reviews['genre'].apply(lambda genre: genre_to_numeric[genre])

categorical_fields = ['score_phrase', 'editors_choice', 'release_year', 'release_month', 'release_day', 'genre']

for field in categorical_fields:
    dummies = pd.get_dummies(reviews[field], prefix=field, drop_first=False)
    reviews = pd.concat([reviews, dummies], axis=1) # concat the new columns

data = reviews.drop(categorical_fields, axis=1)

quant_features = ['score']

for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    data.loc[:, each] = (data[each] - mean)/std

print(data.head())

data.to_csv('data_scaled.csv', index=False)
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
