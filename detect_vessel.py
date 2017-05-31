from skimage import data, io, filters, feature, morphology, measure
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import numpy as np
import pickle
from DecisionModel import Cart, Picture_decision


def get_image(path, asgrey=True, _flatten=False):
    return io.imread(path, as_grey=asgrey, flatten=_flatten)


class vessel_processing:
    def __init__(self):
        pass

    def load_model(self, file):
        with open(file, 'rb') as pickle_file:
            self.clf = pickle.load(pickle_file)
        pass

    def detect(self, file, tile_size, model_list):
        self.picture = Picture_decision(file, None, tile_size)
        self.picture.build_decision_model()
        shape = self.picture.picture.shape
        self.result = np.ndarray(shape=(shape[0], shape[1]), dtype='float')
        i = 0
        for model in model_list:
            self.load_model(model)
            # for decision_x in self.picture.decision_X:
            pixel = self.clf.predict_pixel(self.picture.decision_X[1])
            self.result = np.reshape(pixel, (shape[0], shape[1]))
        io.imsave("obrazek2.png", self.result)
        print("image")

    def save_result(self):
        pass


if __name__ == "__main__":
    vessel = vessel_processing()
    # vessel.load_model("simple_CART_model")
    vessel.detect("Train/input/im01.ppm", tile_size=5, model_list=["simple_CART_model"])
    # vessel.save_result("RESULT.ppm")
