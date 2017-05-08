from skimage import data, io, filters, feature, morphology, measure
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import numpy as np
import pickle

def get_image(path, asgrey=True, _flatten=False):
    return io.imread(path, as_grey=asgrey, flatten=_flatten)

class vessel_processing:
    def __init__(self):
        pass

    def load_model(self,file):
        self.clf = pickle.load(file)
        pass

    def load_image(self, path):
        #Todo in RGB
        self.image = get_image(path)

    def detect(self):
        pass

    def save_result(self):
        pass


if __name__ == "__main__":
    pass
