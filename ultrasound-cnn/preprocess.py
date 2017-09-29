import numpy
import os
from PIL import Image

data_dir = './data/'
"""
data = []
imageFiles = [data_dir + file for file in os.listdir(data_dir) if file.endswith('png')]
for imageFile in imageFiles:
    im = Image.open(imageFile).convert('LA')
    imageData = [point[0] for point in list(im.getdata())]
    imageDataArr = numpy.asarray(imageData)
    data.append(numpy.reshape(imageDataArr, (1, 360, 360)).astype('float32') / 255)
train_test_separator = int(0.8 * len(data))
numpy.save('./data.npy', data[:train_test_separator])
numpy.save('./test_data.npy', data[train_test_separator:])
"""

labels = []
with open(data_dir + 'data.txt') as labelfile:
    for line in labelfile:
        strlabels = line.split()[1:]
        strlabels[-1] = strlabels[-1][:-1]
        label = []
        for strlabel in strlabels:
            label.append(int(strlabel))
            label[-1] = 1 if label[-1] >= 30 else 0
        labels.append(label)
train_test_separator = int(0.8 * len(labels))
numpy.save('./labels.npy', labels[:train_test_separator])
numpy.save('./test_labels.npy', labels[train_test_separator:])