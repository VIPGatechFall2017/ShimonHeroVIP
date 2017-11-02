# ShimonHeroVIP
Vertically Integrated Projects (VIP) at GT for students working on ShimonHero

## Data Collection
We collected our data in the lab using an ultrasound sensor and video recording on a mac. We used our own forearms for data collection. For all of our sessions, we collected data on finger extension and flexion which resulted in a movie recording of the ultrasound images and a ```data.txt``` file with the following format:
```
Thumb Flexion, Pointer Flexion, Middle Flexion, Ring Flexion, Pinky Flexion
```

We only recorded data for one finger being flexed at a time. So, for example, the data point for thumb flexion would be ```99,0,0,0,0``` showing that the thumb was fully flexed while other fingers were not. We also collected data on no fingers flexed which would look like ```0,0,0,0,0```.

After each session, the movie recording and the ```data.txt``` file were saved while recording the data collector's name, the date and time, and the folder location for the session's data.

## Data Processing
Since we stored a video recording of the ultrasound images, the first part of our preprocessing consisted of splitting the movie file into its jpeg frames. We created a python script that converted the movie recording to its frames using the [opencv library](https://docs.opencv.org/3.3.0/index.html). Each 5-minute session was split into 3,600 frames.

We created another python script to take these images and resize them to ```128x128``` or ```28x28``` (depending on the model we're testing). These images were then converted to numpy arrays and stored into two arrays for the training and testing datasets (80/20 split).

The finger flexion data was converted to a vector of size six using one-hot encoding. The first five indices corresponded to each finger where a ```1``` indicated that that specific finger was extended. A ```1``` at the last index indicated that no fingers were extended. This encoded data was also stored into two arrays for the training and testing classification labels.

## Model
### Classifier Type - CNN
To account for all possible combinations of the 6 individual finger states (all five fingers or no fingers), a Convolutional Neural Network fits very well for the task at hand. 

### Convolution
For our model we used two 2D Convolutions, one with 16 filters and another with 8 filters. Both used kernal demensions of 4x4 and a Rectifier activation function.

!(http://colah.github.io/posts/2014-07-Understanding-Convolutions/img/RiverTrain-ImageConvDiagram.png)

### Pooling
After each convolution we downsized the results uding a max pooling size of 2x2. This helps our model train faster and avoid overfitting.

!(https://qph.ec.quoracdn.net/main-qimg-8afedfb2f82f279781bfefa269bc6a90)

### Dropout
We used a Dropout layer of .1 to minimize further overfitting and to help our model train faster.

!(https://cambridgespark.com/content/tutorials/convolutional-neural-networks-with-keras/figures/drop.png)

### Model code
```python
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import numpy
import os
from keras import backend as K 
K.set_image_dim_ordering('th')

images = numpy.load('./data.npy')
labels = numpy.load('./labels.npy')

num_classes = labels.shape[1]
def baseline_model():
    model = Sequential()
    model.add(Conv2D(16, (4, 4), input_shape=(1, 128, 128), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(8, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model = baseline_model()
model.fit(images, labels, validation_split=0.1, epochs=20, batch_size=200, verbose=2)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
```
