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