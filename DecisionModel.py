from sklearn import tree
import numpy as np
import pickle
from skimage import io


def get_image(path, asgrey=True, _flatten=False):
    return io.imread(path, as_grey=asgrey, flatten=_flatten)


class cart:
    '''we use it to create model, that will predict if a pixel is a vessel'''

    def __init__(self, depth):
        self.clf = tree.DecisionTreeClassifier(max_depth=depth)
        # self.clf = self.clf.fit( INPUT, DECISION)

    def save_clf(self, file):
        # something with pickle
        pickle.dump(self.clf, file)


class picture_decision:
    def __init__(self, input_file, output_file, size=5):
        self.input = get_image(input_file, asgrey=False)
        self.output = get_image(output_file, asgrey=False)
        self.size = size
        self.decision_X = []
        self.decision_Y = []

    def set_size(self, size):
        self.size = size

    def build_decision_model(self):
        # several decision models
        self.pixel_model()

    def pixel_model(self):
        self.decision_X.append([])
        self.decision_Y.append([])
        scale = len(self.output) / len(self.input)
        for i, row in enumerate(self.input):
            for j, pixel in enumerate(row):
                self.decision_X.append(pixel)
                self.decision_Y.append(self.output[i*scale][j*scale])

    def pixel_average_model(self):
        average_color_per_row = np.average(self.input, axis=0)
        self.average_color = np.average(average_color_per_row, axis=0)
        self.decision_X.append([])
        self.decision_Y.append([])
        scale = len(self.output) / len(self.input)
        for i, row in enumerate(self.input):
            for j, pixel in enumerate(row):
                decision_x = [pixel, self.average_color]
                self.decision_X.append(decision_x)
                self.decision_Y.append(self.output[i * scale][j * scale])
