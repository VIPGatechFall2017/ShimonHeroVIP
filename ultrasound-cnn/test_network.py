import sys
import numpy
from keras.models import model_from_json

def test_network(model_path, weights_path):
    with open(model_path) as jsonfile:
        model = model_from_json(jsonfile.read())
    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    X = numpy.load('./test_data.npy')
    Y = numpy.load('./test_labels.npy')
    scores = model.evaluate(X, Y, verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

if __name__ == "__main__":
    network_name = sys.argv[1]
    model_path = ''.join(['saved_models/', network_name, '.json'])
    weights_path = ''.join(['saved_models/', network_name, '.h5'])
    test_network(model_path, weights_path)