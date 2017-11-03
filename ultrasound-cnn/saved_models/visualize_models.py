from keras.utils import plot_model
from keras.models import model_from_json
import sys
import os

filenames = [fn for fn in os.listdir('.') if fn.endswith('.json')]
for model_fn in filenames:
    with open(model_fn) as model_file:
        model_json = model_file.read()
        model = model_from_json(model_json)
        plot_model(model, './%s.png' % model_fn[:model_fn.index('.')], show_shapes=True)