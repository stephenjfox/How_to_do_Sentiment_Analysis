import pandas as pd

# Easy is to map a review to simply "positive" or "negative"
# Hard maps the reviews to their full descriptions:
# Amazing, Great, Good, Mediocre, painful, or awful
difficulty = { 'easy': { 'output_nodes': 2 }, 'hard': { 'output_nodes': 6 } }

# Data Preparation
## 0. Read data
data = pd.read_csv('ign.csv')

print('Starting size =', len(data))
print('Starting columns', data.columns)
## 1. Clean data (i.e. remove redundancies)

### this is selecting distinct rows, based on title
### Games rarely get different scores on different platforms, and I don't want
### the rarity to scew the results
data = data.drop_duplicates(subset=['title'])

### the title should be irrelevant. There's no NLP to help that along
data = data.drop(['title'], axis=1)

### the same goes for the platform - PS4 games do just as well as XBone games
data = data.drop(['platform'], axis=1)

# let's see how I did
print('Remaining columns:', data.columns)
print('Post trimming size =', len(data))

## 2. Transform
### Normalize
### * -> Numeric

## 3. Reduction
### Dimensionality Reduction
### Feature Extraction
