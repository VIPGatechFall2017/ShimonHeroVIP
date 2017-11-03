import os
import shutil
if not os.path.exists('combined-data'):
    os.mkdir('combined-data')
frameCount = 0
labels = []
open('combined-data/data.txt', 'w').close()
for i in range(7):
    if i == 0:
        dirname = 'hand-images'
    else:
        dirname = 'hand (%d)-images' % i
    imageFiles = [os.path.join(dirname, filename) for filename in sorted(os.listdir(dirname))]
    for image in imageFiles:
        imagePath = 'combined-data/frame%d.jpg' % frameCount
        shutil.copy2(image, imagePath)
        frameCount += 1
    if i == 0:
        labelname = 'data.txt'
    else:
        labelname = 'data (%d).txt' % i
    with open(labelname) as labelfile:
        with open('combined-data/data.txt', 'a') as combinedfile:
            combinedfile.write(labelfile.read())