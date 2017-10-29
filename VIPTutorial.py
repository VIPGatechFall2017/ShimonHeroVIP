import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(768, input_dim=num_pixels, kernal_initialization='normal', activation='sigmoid'))
	model.add(Dropout(0.2))
	model.add(Dense(512, kernal_initialization='normal', activation='sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(128, kernal_initialization='normal', activation='sigmoid'))
	model.add(Dense(10, kernal_initialization='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))


#Final Accuracy after 10 epochs: 0.9810
#Final Accuracy with default kernal_initialization: 0.9808
#Final Accuracy with rectifier activation instead of sigmoid: 0.9895
#Final Accuracy with rectifier activation and only two dense layers: 0.9982




