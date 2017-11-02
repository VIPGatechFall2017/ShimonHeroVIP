# ShimonHeroVIP
Vertically Integrated Projects (VIP) at GT for students working on ShimonHero

## Data Collection
We collected our data in the lab using an ultrasound sensor and video recording on a mac. We used our own forearms for data collection. For all of our sessions, we collected data on finger extensions which resulted in a movie recording of the ultrasound images and a data.txt file with the following format:
```
Thumb Extension, Pointer Extension, Middle Extension, Ring Extensions, Pinky Extension
```

We only recorded data for one finger being extended at a time. So for thumb extensions the data point would be ```99,0,0,0,0``` showing that the thumb was fully extended while other fingers were not. We also collected data on no finger extended which would like ```0,0,0,0,0```.

After each session, the movie recording and the data.txt file were saved while recording the data collector's name, the date and time, and the folder location for session's data.

## Data Processing
Since we stored a video recording of the ultrasound images, the first part of our preprocessing consisted of splitting the movie file into its jpeg frames. We created a python script that converted the movie recording to its frames using the [opencv library](https://docs.opencv.org/3.3.0/index.html). Each 5-minute session was split into 3,600 frames.

We created another python script to take these images and resize them to 128x128 or 28x28 (depending on the model we're testing). These images were then converted to numpy arrays and stored into two arrays for the training and testing datasets (80/20 split). The finger extension data was converted to a vector of size six using one-hot encoding. The first five indices corresponded to each finger where a ```1``` indicated that that specific finger was extended. A ```1``` at the last index indicated that no fingers were extended. This encoded data was also stored into two arrays for the training and testing classification labels.
