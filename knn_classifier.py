import numpy as np
from data_processing import create_dataset

# constants
directory_name = "image_data"
num_imgs = 12
input_shape = (360, 360, 3)
test_data = []
k = 5


(X_train, X_test, y_train, y_test) = create_dataset(directory_name, num_imgs,
                                                    input_shape, test_data)


def train(X_train, y_train):
    return


def predict(X_train, y_train, x_test, k):
    distances = []
    targets = []

    for i in range(len(X_train)):
        # compute euclidean distance between the observation and all of the
        # data points in the training set
        distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
        print(distance)

        # add it to the list of distances
        distances.append(distance)

    # sort the list so that it will be easy to find each data points nearest
    # neighbors
    distances = sorted(distances)


predict(X_train, y_train, X_test[1, :], k)
