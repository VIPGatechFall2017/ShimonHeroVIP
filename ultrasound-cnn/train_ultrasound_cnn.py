from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import numpy
from PIL import Image
import os
from keras import backend as K 
K.set_image_dim_ordering('th')

images = []
for imagehandle in [item for item in os.listdir('.') if item.endswith('png')]:
    im = Image.open(imagehandle).convert('LA')
    data = [point[0] for point in list(im.getdata())]
    arr = numpy.asarray(data)
    images.append(numpy.reshape(arr, (1, 360, 360)).astype('float32') / 255)

train_test_separator = int(0.7 * len(images))

x_train = images[:train_test_separator]
x_test = images[train_test_separator:]

y_train = images[:train_test_separator]
y_test = images[train_test_separator:]

labels = []
with open('data.txt') as labelfile:
    line = labelfile.readline()
    strlabels = line.split()[1:]
    label = []
    for strlabel in strlabels:
        label.append(int(strlabel[0]))
    labels.append(label)

labels = np_utils.to_categorical(labels)
num_classes = labels.shape[1]
def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (20, 20), input_shape=(1, 360, 360), activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.2))
    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
model = baseline_model()
model.fit(images, labels, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=2)
