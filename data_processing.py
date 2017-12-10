import numpy as np
from scipy.misc import imread
from skimage.color import rgb2grey
from skimage.transform import resize

#
# pass the name of the folder containing the image files into this function
# along with the number of images as an integer, the images' 3D shape as a
# tuple, the desired image shape after conversion to grayscale and downsampling
# and the name of the file containing the raw data to get a tuple of the
# format (X_train, X_test, y_train, y_test)
def create_dataset(directory_name, num_imgs, input_shape, resize_shape,
                   crop_amounts, test_data_filename):
    dataset = load_and_process_data(directory_name, num_imgs, input_shape,
                                    resize_shape, crop_amounts)
    test_data = load_and_process_test_data(test_data_filename)
    return make_training_and_testing_sets(dataset, num_imgs, test_data)


def load_and_process_data(directory_name, num_imgs, input_shape, resize_shape,
                          crop_amounts):
    # create an empty array with an appropriate 4D shape for the input images
    dataset_shape = (num_imgs,) + input_shape
    dataset = np.zeros(shape=dataset_shape)

    # iterate over files in the image_data folder and load them, skipping
    # files that aren't images
    for i in range(0, num_imgs):
        dataset[i] = imread("%s/frame%s.jpg" % (directory_name, i))

    # scale each entry in the dataset by 255 to get each value between 0 and 1
    dataset = dataset / 255

    # crop the extra black space off of the sides of the images
    new_dataset = np.zeros(shape=(num_imgs,) + (input_shape[0],
                                                input_shape[1]
                                                - crop_amounts[0]
                                                - crop_amounts[2],
                                                input_shape[2]))
    for i in range(0, num_imgs):
        new_dataset[i] = crop(dataset[i], 45, 0, 60, 0)

    dataset = new_dataset

    # determine the new shape of the dataset in preparation for downsampling
    # the images, iterate over the images, converting to grayscale and
    # downsampling, and then replace the old dataset with the new one
    new_shape = (num_imgs,) + resize_shape
    new_dataset = np.zeros(new_shape)
    for i in range(0, num_imgs):
        new_dataset[i] = convert_to_grayscale_and_downsample(dataset[i],
                                                             resize_shape)
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


def convert_to_grayscale_and_downsample(image, resize_shape):
    # convert to grayscale
    image = rgb2grey(image)

    # downsample image to 28x28
    image = resize(image, resize_shape, preserve_range=True)

    return image


def crop(image, left_off, top_off, right_off, bottom_off):
    image = image[top_off:image.shape[0] - bottom_off,
            left_off:image.shape[1] - right_off,]
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
directory_name = "image_data/1/hand-images"
num_imgs = 5
input_shape = (740, 600, 3)
resize_shape = (28, 28)
crop_amounts = (45, 0, 60, 0) # left, top, right, bottom
test_data_filename = "data.txt"


(X_train, X_test, y_train, y_test) = create_dataset(directory_name, num_imgs,
                                                    input_shape, resize_shape,
                                                    crop_amounts,
                                                    test_data_filename)
