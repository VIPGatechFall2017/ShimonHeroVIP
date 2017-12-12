import numpy as np
from scipy.misc import imread
from skimage.color import rgb2grey
from skimage.transform import resize


# pass the name of the folder containing the image files into this function
# along with the number of images as an integer, the images' 3D shape as a
# tuple, the desired image shape after conversion to grayscale and downsampling
# and the name of the file containing the raw data to get a tuple of the
# format (X_train, X_test, y_train, y_test)
def create_dataset(directory_name, file_name, num_imgs, input_shape,
                   resize_shape, crop_amounts, test_data_filename):
    dataset = load_and_process_data(directory_name, file_name, num_imgs,
                                    input_shape, resize_shape, crop_amounts)
    test_data = load_and_process_test_data(test_data_filename)
    return make_training_and_testing_sets(dataset, num_imgs, test_data,
                                          resize_shape)

# loads the images one at a time, performing any preprocessing and then adding
# them to the returned dataset
def load_and_process_data(directory_name, file_name, num_imgs, input_shape,
                          resize_shape, crop_amounts):
    # create an empty array with an appropriate 4D shape for the input images
    dataset_shape = (num_imgs,) + resize_shape
    dataset = np.zeros(shape=dataset_shape)

    # iterate over files in the image_data folder and load them, skipping
    # files that aren't images
    for i in range(0, num_imgs):
        image = imread("%s/%s%s.jpg" % (directory_name, file_name, i))

        # crop the extra black space off of the sides of the image
        image = crop(image, crop_amounts)

        # downsample the image and convert to grayscale
        image = convert_to_grayscale_and_downsample(image, resize_shape)

        # add the image to the dataset
        dataset[i] = image

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

        finger_open = False
        for i in range(len(processed_line)):
            processed_line[i] = 1 if int(processed_line[i]) > 0.5 else 0
            if processed_line[i] == 1:
                finger_open = True

        # adds sixth spot to represent hand with all fingers closed and sets it
        # appropriately
        open_hand = [0]
        if not finger_open:
            open_hand = [1]
        processed_line = processed_line + open_hand
        test_data.append(processed_line)

    # convert to a numpy array
    test_data = np.asarray(test_data)

    return test_data


def convert_to_grayscale_and_downsample(image, resize_shape):
    image = rgb2grey(image)
    image = resize(image, resize_shape, preserve_range=True)
    return image

# crops the amounts passed in off of each of the sides of the image passed in
def crop(image, crop_amounts):
    image = image[crop_amounts[1]:image.shape[0] - crop_amounts[3],
            crop_amounts[0]:image.shape[1] - crop_amounts[2], ]
    return image

# shuffles the training and testing sets in unison and returns them in a tuple
def make_training_and_testing_sets(dataset, num_imgs, test_data, resize_shape):
    images_shuffled = np.zeros(dataset.shape)
    test_data_shuffled = np.zeros(test_data.shape)
    indices = range(num_imgs)
    np.random.shuffle(indices)
    for i in indices:
        images_shuffled[i] = dataset[i]
        test_data_shuffled[i] = test_data[i]

    # reshapes the image array to have 4 dimensions so that the first Conv2D
    # layer will accept it
    images_shuffled = images_shuffled.reshape((num_imgs, resize_shape[0],
                                               resize_shape[1], 1))

    return (images_shuffled, test_data_shuffled)


# for testing
# directory_name = "image_data/1/hand-images"
# file_name = "frame"  # whatever text comes before the number in the names of
# the image files (assumes files are jpg)
# num_imgs = 5
# input_shape = (740, 600, 3)
# resize_shape = (40, 40)
# crop_amounts = (45, 0, 60, 0)  # left, top, right, bottom
# test_data_filename = "data.txt"
#
# (images, test_data) = create_dataset(directory_name, file_name,
#                                                     num_imgs, input_shape,
#                                                     resize_shape,
#                                                     crop_amounts,
#                                                     test_data_filename)