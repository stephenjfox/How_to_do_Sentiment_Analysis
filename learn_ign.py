import tflearn
from tflearn.data_utils import load_csv, to_categorical

# might need (..., categorical_labels=True, n_classes=11)
data, labels = load_csv('data_scaled.fix.csv', target_column=0)

total_records = (len(data) + len(labels)) // 2 # incase things got funny - will have to compensate later

# Training on 70% of the data
training_size = int(0.7 * total_records)
validation_portion = 0.1 # will use later

# Split our data into proportional chunks
trainX, trainY = data[:training_size], labels[training_size:]
testX, testY = data[:training_size], labels[training_size:]

# convert to one-hot?
trainY = to_categorical(trainY, nb_classes=11)
testY = to_categorical(testY, nb_classes=11)

# The Network

# as many inputs as there are columns... I don't know how to pick this..
number_of_inputs = len(trainX[0])

net = tflearn.input_data([None, number_of_inputs])
net = tflearn.embedding(net, input_dim=number_of_inputs**2, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 11, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.01,
                        loss='categorical_crossentropy')

model = tflearn.DNN(net)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
            batch_size=48)

# Load into TFLearn model: http://tflearn.org/data_utils/#load_csv
model.save('study_ign.tflearn')
