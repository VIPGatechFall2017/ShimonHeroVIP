import sys
from keras.models import model_from_json
import numpy
from keras import backend as K 
K.set_image_dim_ordering('th')

model_name = sys.argv[1]
with open(''.join(['saved_models/', model_name, '.json'])) as modelfile:
    model = model_from_json(modelfile.read())
images = numpy.load('./data.npy')
labels = numpy.load('./labels.npy')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(images, labels, validation_split=0.2, epochs=100, batch_size=200, verbose=2)
# serialize weights to HDF5
model.save_weights("model.h5")