import numpy as np
from process3 import create_dataset
from sklearn.metrics import accuracy_score


def train(X_train, y_train):
    return


def predict(X_train, y_train, x_test, k):
    distances = []
    targets = []

    for i in range(len(X_train)):
        # compute distance (i.e. euclidean, etc.) between the observation and all of the
        # data points in the training set
        
        distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :]))) #euclidean
        # distance = np.sum(np.absolute((x_test - X_train[i, :]))) #manhattan    
        # distance = 1 - np.dot(x_test.flatten(), X_train[i, :].flatten()) / (np.sqrt(np.sum(np.square(x_test))) * np.sqrt(np.sum(np.square(X_train[i, :])))) #cosine distance
        # distance = 0.5 * np.sum((x_test - X_train[i, :])**2 / (x_test + X_train[i, :] + 0.000001))

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
    target = None
    max_count = 0
    for i in range(len(unique_targets)):
        if counts[i] > max_count:
            target = unique_targets[i]
            max_count = counts[i]
    #return counts_to_targets[np.amax(counts)]
    return target


def knn(X_train, y_train, X_test, predictions, k):
    # train on the input data
    train(X_train, y_train)

    # loop over all observations
    for i in range(len(X_test)):
        predictions.append(predict(X_train, y_train, X_test[i, :], k))

def main():

    k_array = [1, 50, 100, 300, 1000]
    for k_val in k_array:
        directory_name = "skywalkerKippFrames_1"
        num_imgs = 3600
        input_shape = (360, 360, 3)
        resize_shape = (28, 28)  
        test_data_filename = "skywalker_data/Kipp_10_4/1/data.txt"  
        k = k_val
        print("K = " + str(k))
        predictions = []


        (X_train, X_test, y_train, y_test) = create_dataset(directory_name, num_imgs,
                                                        input_shape, resize_shape,
                                                        test_data_filename)

        knn(X_train, y_train, X_test, predictions, k)
        predictions = np.asarray(predictions)
        accuracy = accuracy_score(y_test, predictions)
        print("\nThe accuracy of our classifier is {0}%".format(accuracy*100))
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

if __name__ == "__main__": main()

