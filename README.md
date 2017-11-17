# ShimonHeroVIP
Vertically Integrated Projects (VIP) at GT for students working on ShimonHero

## Problem Introduction
Our task was to design a convolutional neural network to classify finger flexion given ultrasound data of an arm. Each datapoint is a raw ultrasound image and a corresponding label representing which finger was flexed when the image was taken. We tried a plethora of different architectures with varying degrees of results. We trained our networks on data from one individual, due to the high variance of the images between arms. Our training accuracy reached as high as 80%, but even the simplest networks suffered from severe overfitting.
![intro_image](https://github.com/VIPGatechFall2017/ShimonHeroVIP/blob/Jared/results/frame1.jpg)

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

![conv image](http://colah.github.io/posts/2014-07-Understanding-Convolutions/img/RiverTrain-ImageConvDiagram.png)

### Pooling
After each convolution we downsized the results uding a max pooling size of 2x2. This helps our model train faster and avoid overfitting.

![pooling image](https://qph.ec.quoracdn.net/main-qimg-8afedfb2f82f279781bfefa269bc6a90)

### Dropout
We used a Dropout layer of .1 to minimize further overfitting and to help our model train faster.

![dropout image](https://cambridgespark.com/content/tutorials/convolutional-neural-networks-with-keras/figures/drop.png)

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

## Results
The training had rather modest results. We had major struggles with overfitting, as no matter what we did to the network and what the training was doing, the test accuracy would not increase. Going forward, we should attempt to identify underlying issues of the architecture, as decreasing its copmlexity proved to be mostly ineffective. Some of the models we tried are shown below.

![model1](https://github.com/VIPGatechFall2017/ShimonHeroVIP/blob/Jared/results/Model%201.png)
![model1-a](https://github.com/VIPGatechFall2017/ShimonHeroVIP/blob/Jared/results/Model%201%20-%20Analysis.jpg)
Here, we had a large input size (128x128), so the network was trained for only 20 epochs. Right away, the overfitting can be seen. The training accuracy approaches 80% but the testing accuracy stays at approximately random guessing performance. Therefore, the model must be too complex as it doesn't generalize well. We started to combat this by reducing input size.

![model2](https://github.com/VIPGatechFall2017/ShimonHeroVIP/blob/Jared/results/Model%202.png)
![model2-a](https://github.com/VIPGatechFall2017/ShimonHeroVIP/blob/Jared/results/Model%202%20-%20Analysis.jpg)
Given the smaller input size, we decided to train for 200 epochs rather than 20. The training accuracy was actually better than in the previous network, but the testing accuracy was still random. Therefore, more adjustments needed to be made.

![model3](https://github.com/VIPGatechFall2017/ShimonHeroVIP/blob/Jared/results/Model%203.png)
![model3-a](https://github.com/VIPGatechFall2017/ShimonHeroVIP/blob/Jared/results/Model%203%20-%20Analysis.jpg)
Here, we are actually able to see the test data start to follow the curve of the training data, but we have unfortunately been unable to replicate these results. We plan on using this network as a starting point going forward to try to get the improve the testing data.

![model4-a](https://github.com/VIPGatechFall2017/ShimonHeroVIP/blob/Jared/results/Model%204%20-%20Analysis.jpg)
Model 4 is the same as model 3, but we moved to a larger dataset (3 sessions of data to 7). This data change also changed people, which is likely why the larger dataset actually performed worse on the same model. From this point forward, we used the larger dataset.

![model5](https://github.com/VIPGatechFall2017/ShimonHeroVIP/blob/Jared/results/Model%205.png)
![model5-a](https://github.com/VIPGatechFall2017/ShimonHeroVIP/blob/Jared/results/Model%205%20-%20Analysis.jpg)

The training accuracies went down significantly in model 4, so for this model we considered increasing the convolution layer size. As expected, this slightly improved training accuracy, but obviously did not help testing accuracy.

![model6](https://github.com/VIPGatechFall2017/ShimonHeroVIP/blob/Jared/results/Model%206.png)
![model6-a](https://github.com/VIPGatechFall2017/ShimonHeroVIP/blob/Jared/results/Model%206%20-%20Analysis.jpg)

In this model, an entire convolutional layer was removed to try to combat overfitting. The attempt was unsuccessful.

![model7](https://github.com/VIPGatechFall2017/ShimonHeroVIP/blob/Jared/results/Model%207.png)
![model7-a](https://github.com/VIPGatechFall2017/ShimonHeroVIP/blob/Jared/results/Model%207%20-%20Analysis.jpg)

Decreasing the number of filters from 24 to 8 in the convolutional layer also had little effect on overfitting. We will be looking into the cause of this overfitting using the data that we've gathered to try to pinpoint a cause. 

## Followup : Data Shuffling
Upon further investigation, it was discovered that the data was not properly shuffled before training the above networks. In order to test whether this was having an effect, Model 3 was retrained on shuffled data. The results can be seen below.

![model3-shuffled](https://github.com/VIPGatechFall2017/ShimonHeroVIP/blob/Jared/results/model3-shuffled.jpg)

This shows a clear improvement over the previous performance, indicating that the issue which was hindering performance was an issue with data, not an issue with overfitting/model complexity. This model trained relatively well. In order to test whether the testing accuracy issues could be entirely attributed to a lack of shuffling rather than overfitting, Model 2 was retrained on the new shuffled data. Model 2 is more complex than model 3 and takes an input of 48x48 patches. The results can be seen below.

![model2-shuffled](https://github.com/VIPGatechFall2017/ShimonHeroVIP/blob/Jared/results/model2-shuffled.jpg)

With model 2, up to 70% testing accuracy was acheived, with no significant signs of overfitting, despite the complexity of the network. One interesting fact is that toward the beginning of training, the testing accuracy is higher than the training accuracy. According to discussions on the tensorflow source, this can occur at the early stages of training when using dropout layers, which this network uses.

Based on these results, we can conclude that the poor performance of the networks previously was due to not shuffling the data properly.
