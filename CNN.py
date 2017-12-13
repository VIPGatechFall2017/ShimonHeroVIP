from data_processing import create_dataset
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D


# constants
directory_name = "image_data/1/hand-images"
file_name = "frame" # whatever text comes before the number in the names of the
                    # image files (assumes files are jpg)
num_imgs = 3600
input_shape = (740, 600, 3)
resize_shape = (40, 40)
crop_amounts = (45, 0, 60, 0) # left, top, right, bottom
test_data_filename = "data.txt"
num_classes = 6


(images, test_data) = create_dataset(directory_name, file_name,
                                                    num_imgs, input_shape,
                                                    resize_shape, crop_amounts,
                                                    test_data_filename)


def build_model():
    model = Sequential()
    model.add(Conv2D(16, (5, 5), input_shape=(40, 40, 1),  activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.1))
    model.add(Conv2D(8, (4, 4), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.1))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])
    return model


model = build_model()


model.fit(images, test_data, validation_split=0.2, epochs=200, batch_size=250,
          verbose=2)