import pandas as pd

# Easy is to map a review to simply "positive" or "negative"
# Hard maps the reviews to their full descriptions:
# learned more about the data set: "Masterpiece", "Unbearable", "Disaster", "Okay" are also rating phrases
# Masterpiece, Amazing, Great, Good, Okay, Mediocre, Awful, Painful, Unbearable, Disaster

difficulty = { 'easy': { 'output_nodes': 2 }, 'hard': { 'output_nodes': 10 } }

# Data Preparation
## 0. Read data
## 1. Clean data (i.e. remove redundancies)
data = pd.read_csv('cleaned_ign_reviews.csv')


# Data Investigation
## get one of each phrased review
one_of_each = data.drop_duplicates(['score_phrase'])
distincts_as_reviews = one_of_each[['title', 'score_phrase', 'score']]
print(distincts_as_reviews.sort_values('score', ascending=False))

# end investigation

## 2. Transform
### Normalize

### * -> Numeric

## 3. Reduction
### Dimensionality Reduction
### Feature Extraction
