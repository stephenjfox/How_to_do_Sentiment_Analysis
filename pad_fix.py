import pandas as pd
from ast import literal_eval
from tflearn.data_utils import pad_sequences

df = pd.read_csv('titles_as_vectors.csv')

print("\n----From the file:----\n", type(df['title'][0]))

df['title'] = df['title'].apply(literal_eval)

print("\n----*.apply(literal_eval):----\n", type(df['title'][0]))

padded = pad_sequences(df['title'], maxlen=20)

print("\n----Sequence padded:----\n", type(padded), "\ndata:", padded[0])

print()
print("----Matching up lengths:----\n")

print("df['title'] length =", len(df['title']))
print("padded data length =", len(padded))

# data = [[1, 2, 3]]
#
# padded = pad_sequences(data, maxlen=10)
#
# print(padded)
