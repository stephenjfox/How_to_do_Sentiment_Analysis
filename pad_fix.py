import pandas as pd
from ast import literal_eval
from tflearn.data_utils import pad_sequences

df = pd.read_csv('titles_as_vectors.csv')

print("\n----From the file:----\n", df['title'].head())

df['title'] = df['title'].apply(literal_eval)

print("\n----*.apply(literal_eval):----\n", df['title'].head())

df['title'] = pad_sequences(df['title'], maxlen=20)

print("\n----Sequence padded:----\n", df['title'].head())

# data = [[1, 2, 3]]
#
# padded = pad_sequences(data, maxlen=10)
#
# print(padded)
