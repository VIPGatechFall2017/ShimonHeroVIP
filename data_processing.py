import numpy as np
from PIL import Image
from resizeimage import resizeimage
from scipy.misc import imread
from skimage.color import rgb2grey
from sklearn.preprocessing import MinMaxScaler
from skimage import transform


# pass the name of the folder containing the image files into this function
# along with the number of images as an integer, the images' 3D shape as a
# tuple, the desired image shape after conversion to grayscale and downsampling
# and the name of the file containing the raw data to get a tuple of the
# format (X_train, X_test, y_train, y_test)
def create_dataset(directory_name, num_imgs, input_shape, resize_shape,
                   test_data_filename):
    dataset = load_and_process_data(directory_name, num_imgs, input_shape,
                                    resize_shape)
    test_data = load_and_process_test_data(test_data_filename)
    return make_training_and_testing_sets(dataset, num_imgs, test_data)


def load_and_process_data(directory_name, num_imgs, input_shape, resize_shape):
    # create an empty array with an appropriate 4D shape for the input images
    dataset_shape = (num_imgs,) + input_shape
    dataset = np.zeros(shape=dataset_shape)

    # iterate over files in the image_data folder and load them, skipping
    # files that aren't images
    for i in range(0, num_imgs):
        dataset[i] = imread("%s/hand.mov_%s.png" % (directory_name, i))

    # reshape the data to a 2D array and apply the MinMaxScaler
    num_pixels = input_shape[0] * input_shape[1] * input_shape[2]
    processing_shape = (num_imgs, num_pixels)
    dataset = np.reshape(dataset, newshape=processing_shape)
    scaler = MinMaxScaler()
    scaler.fit(dataset)
    dataset = scaler.transform(dataset)

    # reshapes the dataset back to a 4D array
    dataset = np.reshape(dataset, newshape=dataset_shape)

    # determine the new shape of the dataset in preparation for downsampling
    # the images, iterate over the images, converting to grayscale and
    # downsampling, and then replace the old dataset with the new one
    new_shape = (num_imgs,) + resize_shape
    new_dataset = np.zeros(new_shape)
    for i in range(len(dataset)):
        new_dataset[i] = convert_to_grayscale_and_lower_dim(dataset[i],
                                                            new_shape)
    dataset = new_dataset

    return dataset


def load_and_process_test_data(test_data_filename):
    # open the file containing the data and create an empty list to hold the
    # data
    file = open(test_data_filename, "r")
    test_data = []

    # iterate over the lines in the file, removing unnecessary characters and
    # converting all "class scores" to either 0 or 1 based on whether the
    # finger is "on or off"
    for line in file:
        processed_line = line.split(",")[1].strip(" ;\n").split(" ")

        for i in range(len(processed_line)):
            processed_line[i] = 1 if int(processed_line[i]) > 0.5 else 0

        test_data.append(processed_line)

    return test_data


def convert_to_grayscale_and_lower_dim(image, new_shape):
    # convert to grayscale
    image = rgb2grey(image)

    # convert the numpy array to a PIL image and then resize it to be smaller
    image = Image.fromarray(image)
    new_dim = [new_shape[1], new_shape[2]]
    image = resizeimage.resize_thumbnail(image, new_dim)

    # convert back to a numpy array
    image = np.array(image.getdata()).reshape(tuple(new_dim))

    return image


def make_training_and_testing_sets(dataset, num_imgs, test_data):
    # produce arrays made up of approx. 80% of the input and approx. 20% of the
    # input, respectively
    training_size = int(np.floor(num_imgs * .8))
    X_train = np.array(dataset[0:training_size])
    X_test = np.array(dataset[training_size:len(dataset)])

    # produce arrays containing the data from the text file using length and
    # dimensions of the input arrays
    y_train = np.array(test_data[0:training_size])
    y_test = np.array(test_data[training_size:len(dataset)])

    return (X_train, X_test, y_train, y_test)

# for testing
#
# directory_name = "image_data"
# num_imgs = 150
# input_shape = (360, 360, 3)
# resize_shape = (28, 28)
# test_data_filename = "data.txt"
#
#
# (X_train, X_test, y_train, y_test) = create_dataset(directory_name, num_imgs,
#                                                     input_shape, resize_shape,
#                                                     test_data_filename)
