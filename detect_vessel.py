from skimage import data, io, filters, feature, morphology, measure
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import numpy as np
import pickle
from DecisionModel import Cart, Picture_decision
from statistics import *


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

    def get_tile_score(self, img_inputs, img_outputs, tile_size):
        for input, output in zip(img_inputs, img_outputs):
            output_img = get_image(output, asgrey=True, _flatten=True)

            img = get_image(input, asgrey=False)
            img_x = img.shape[0]
            img_y = img.shape[1]
            border = np.zeros((img_x + (tile_size - 1), img_y + (tile_size - 1), 3), dtype=np.uint8)
            border_padding = int((tile_size - 1) / 2)
            border[border_padding: - border_padding, border_padding: - border_padding,
            :] = img
            io.imsave("tmp_img.ppm", border)
            ### Insert image into frame
            picture_decision = Picture_decision("tmp_img.ppm", None, tile_size)

            picture_decision.build_tile_decision_models()
            print("decision models built")
            ### compute decisions and score
            for model_num in range(2, 7):
                for depth in range(3, 16):
                    self.load_model("CART/model_%.2d_%.2d_%.2d" % (model_num, depth, tile_size))
                    pixel = self.clf.predict_pixel(picture_decision.decision_X[model_num])
                    output_decision = np.reshape(pixel, (img_x, img_y))
                    print(get_score(output_decision, output_img),
                          "CART/model_%.2d_%.2d_%.2d" % (model_num, depth, tile_size))
                    # border = OBRAZEK OBOK OBRAZKA NA JEDNYM
                    # io.imsave("RESULTS/model_%.2d_%.2d_%.2d.png" % (model_num, depth, tile_size), self.result)
                    io.imsave("RESULTS/model_%.2d_%.2d_%.2d.png" % (model_num, depth, tile_size), output_decision)

    def get_pixel_score(self, img_inputs, img_outputs):
        tile_size = 3
        for input, output in zip(img_inputs, img_outputs):
            output_img = get_image(output, asgrey=True, _flatten=True)

            picture_decision = Picture_decision(input, None, tile_size)
            img_x = picture_decision.picture.shape[0]
            img_y = picture_decision.picture.shape[1]
            picture_decision.build_pixel_decision_models()
            ### compute decisions and score
            for model_num in range(0, 2):
                for depth in range(3, 16):
                    self.load_model("CART/model_%.2d_%.2d_%.2d" % (model_num, depth, tile_size))
                    pixel = self.clf.predict_pixel(picture_decision.decision_X[model_num])
                    output_decision = np.reshape(pixel, (img_x, img_y))
                    print(get_score(output_decision, output_img),
                          "CART/model_%.2d_%.2d_%.2d" % (model_num, depth, tile_size))
                    """ DAJE SCORE DLA OBRAZKA ALE ICH NIE DODAJE. POPRAWIC!"""
                    io.imsave("RESULTS/model_%.2d_%.2d_%.2d.png" % (model_num, depth, tile_size), output_decision)


if __name__ == "__main__":
    vessel = vessel_processing()
    vessel.get_pixel_score(["Test/input/im35.ppm"], ["Test/output/m_im35.ppm"])
    # vessel.get_pixel_score(["Train/input/im01.ppm"], ["Train/output/m_im01.ppm"])
    #
    for tile_size in range(3, 27, 2):
        vessel.get_tile_score(["Test/input/im35.ppm"], ["Test/output/m_im35.ppm"], tile_size)
        # vessel.get_tile_score(["Test/im21.ppm"], ["Test/m_im21.ppm"], tile_size)
        # vessel.load_model("simple_CART_model")
        # vessel.detect("Train/input/im01.ppm", tile_size=5, model_list=["simple_CART_model"])
        # vessel.save_result("RESULT.ppm")
