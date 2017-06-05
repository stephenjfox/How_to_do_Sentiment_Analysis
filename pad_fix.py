from tflearn.data_utils import pad_sequences

data = [[1, 2, 3]]

padded = pad_sequences(data, maxlen=10)

print(padded)
