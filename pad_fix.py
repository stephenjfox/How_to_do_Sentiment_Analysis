import pandas as pd
from ast import literal_eval
from tflearn.data_utils import pad_sequences

df = pd.read_csv('titles_as_vectors.csv')

print("\n----From the file:----\n", type(df['title'][0]))

df['title'] = df['title'].apply(literal_eval)

print("\n----*.apply(literal_eval):----\n", type(df['title'][0]))

padded = pad_sequences(df['title'], maxlen=20)

print("\n----Sequence padded:----\n", type(padded), "\ndata:", padded[0])

def cheap_pad(array, length, pad_with=0):
    """
    Performs a padding of the array elements in place
    """
    if len(array) < length:
        while len(array) < length:
            array.append(pad_with)
    if len(array) > length:
        return array[:length]
    return array

print()
print("----Matching up lengths:----\n")

df['title'] = df['title'].apply(lambda array: cheap_pad(array, 15))

print("Updated df['title']")
print(df['title'].head())

df.to_csv('ign_formatted_padded.csv')
# data = [[1, 2, 3]]
#
# padded = pad_sequences(data, maxlen=10)
#
# print(padded)
