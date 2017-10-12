import numpy as np
from scipy.misc import imread
from os import listdir
from sklearn.preprocessing import MinMaxScaler


# pass the name of the folder containing the image files into this function
# along with the number of images as an integer and the images' 3D shape as a
# tuple to get a tuple of the format (X_train, X_test, y_train, y_test)
def create_dataset(directory_name, num_imgs, input_shape):
    dataset = load_and_process_data(directory_name, num_imgs, input_shape)
    return make_training_and_testing_sets(dataset, num_imgs)


def load_and_process_data(directory_name, num_imgs, input_shape):
    # creates an empty array with an appropriate 4D shape for the input images
    dataset_shape = (num_imgs,) + input_shape
    dataset = np.zeros(shape=dataset_shape)

    # iterates over files in the image_data folder and loads them, skipping
    # files that aren't images
    i = 0
    for filename in listdir(directory_name):
        try:
            dataset[i] = imread("%s/%s" % (directory_name, filename))
            i += 1
        except IOError:
            print("Unable to load data from %s" % filename)

    # reshapes the data to a 2D array and applies the MinMaxScaler
    num_pixels = input_shape[0] * input_shape[1] * input_shape[2]
    processing_shape = (num_imgs, num_pixels)
    dataset = np.reshape(dataset, newshape=processing_shape)
    scaler = MinMaxScaler()
    scaler.fit(dataset)
    dataset = scaler.transform(dataset)

    # reshapes the dataset back to a 4D array
    dataset = np.reshape(dataset, newshape=dataset_shape)

    return dataset


def make_training_and_testing_sets(dataset, num_imgs):
    # produce arrays made up of approx. 80% of the input and approx. 20% of the
    # input, respectively
    training_size = int(np.floor(num_imgs * .8))
    X_train = np.array(dataset[0: training_size])
    X_test = np.array(dataset[training_size: len(dataset)])

    # produce output arrays using length and dimensions of the input arrays
    y_train = np.zeros(shape=X_train.shape)
    y_test = np.zeros(shape=X_test.shape)

    return (X_train, X_test, y_train, y_test)
