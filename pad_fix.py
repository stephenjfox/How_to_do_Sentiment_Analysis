import pandas as pd

from tflearn.data_utils import pad_sequences

df = pd.read_csv('titles_as_vectors.csv')

df['title'] = pad_sequences(df['title'])

print(df['title'][0])
