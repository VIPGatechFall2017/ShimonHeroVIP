import numpy as np
from collections import Counter
from data_processing import create_dataset


# constants
directory_name = "image_data"
num_imgs = 10
input_shape = (360, 360, 3)
test_data_filename = "data.txt"
k = 5


(X_train, X_test, y_train, y_test) = create_dataset(directory_name, num_imgs,
                                                    input_shape,
                                                    test_data_filename)


def train(X_train, y_train):
    return


def predict(X_train, y_train, x_test, k):
    distances = []
    targets = []

    for i in range(len(X_train)):
        # compute euclidean distance between the observation and all of the
        # data points in the training set
        distance = np.sqrt(np.sum(np.square(x_test - X_train[i])))

        # add it to the list of distances
        distances.append([distance, i])

    # sort the list so that it will be easy to find each data point's nearest
    # neighbors
    distances = sorted(distances)

    # make a list of the k neighbors' targets
    for i in range(k):
        index = distances[i][1]
        print("%sth index: %s" % (i, index))
        targets.append(y_train[index])

    # return the most common target
    return Counter(targets).most_common(1)[0][0]


predict(X_train, y_train, X_test[1], k)
