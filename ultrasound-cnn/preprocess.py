import numpy
import os
from PIL import Image
import scipy.misc

data_dir = './data/'
data = []
imageFiles = [data_dir + file for file in os.listdir(data_dir) if file.endswith('png')]
for i, imageFile in enumerate(imageFiles):
    im = Image.open(imageFile).convert('LA')
    im = im.resize((28, 28), Image.ANTIALIAS)
    if i % 1000 == 0: im.show()
    imageData = [point[0] for point in list(im.getdata())]
    imageDataArr = numpy.asarray(imageData)
    data.append(numpy.reshape(imageDataArr, (1, 28, 28)).astype('float32') / 255)

labels = []
with open(data_dir + 'data.txt') as labelfile:
    for i, line in enumerate(labelfile):
        strlabels = line.split()[1:]
        strlabels[-1] = strlabels[-1][:-1]
        rawvalues = [int(strlabel) for strlabel in strlabels]
        label = [0] * 6
        if not any([rawvalue >= 30 for rawvalue in rawvalues]):
            label[5] = 1
        else:
            label[rawvalues.index(max(rawvalues))] = 1
        if i % 1000 == 0: print(label)
        labels.append(label)

categories = [[i for i in range(len(labels)) if labels[i][val] == 1] for val in range(6)]
categoryCounts = [len(category) for category in categories]
print(categoryCounts)
minCategory = min(categoryCounts)
balanced_data = []
balanced_labels = []
for i in range(minCategory):
    for category in categories:
        balanced_data.append(data[category[i]])
        balanced_labels.append(labels[category[i]])
indices = range(minCategory)
numpy.random.shuffle(indices)
data = [balanced_data[index] for index in indices]
labels = [balanced_labels[index] for index in indices]


train_test_separator = int(0.8 * len(data))
numpy.save('./data.npy', data[:train_test_separator])
numpy.save('./test_data.npy', data[train_test_separator:])
numpy.save('./labels.npy', labels[:train_test_separator])
numpy.save('./test_labels.npy', labels[train_test_separator:])