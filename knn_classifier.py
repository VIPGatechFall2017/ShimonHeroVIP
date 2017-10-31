import numpy as np
from data_processing import create_dataset
from sklearn.metrics import accuracy_score

# constants
directory_name = "image_data"
num_imgs = 300
input_shape = (360, 360, 3)
resize_shape = (28, 28)
test_data_filename = "data.txt"
k = 2
predictions = []


(X_train, X_test, y_train, y_test) = create_dataset(directory_name, num_imgs,
                                                    input_shape, resize_shape,
                                                    test_data_filename)


def train(X_train, y_train):
    return


def predict(X_train, y_train, x_test, k):
    distances = []
    targets = []

    for i in range(len(X_train)):
        # compute euclidean distance between the observation and all of the
        # data points in the training set
        distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))

        # add it to the list of distances
        distances.append([distance, i])

    # sort the list so that it will be easy to find each data point's nearest
    # neighbors
    distances = sorted(distances)

    # make a list of the k neighbors' targets
    for i in range(k):
        index = distances[i][1]
        targets.append(y_train[index])

    # return the most common target
    unique_targets, counts = np.unique(targets, return_counts=True, axis=0)
    counts_to_targets = dict(zip(counts, unique_targets))
    return counts_to_targets[np.amax(counts)]


def knn(X_train, y_train, X_test, predictions, k):
    # train on the input data
    train(X_train, y_train)

    # loop over all observations
    for i in range(len(X_test)):
        predictions.append(predict(X_train, y_train, X_test[i, :], k))


knn(X_train, y_train, X_test, predictions, k)
predictions = np.asarray(predictions)
accuracy = accuracy_score(y_test, predictions)
print("\nThe accuracy of our classifier is {0}%".format(accuracy*100))
