from sklearn import tree
import numpy as np
import pickle
from skimage import io, measure
from picture import Picture

import warnings
import cv2


def get_image(path, asgrey=True, _flatten=False):
    return io.imread(path, as_grey=asgrey, flatten=_flatten)


def merge(*iters):
    for it in iters:
        yield from it


class Cart:
    """We use it to create model, that will predict if a pixel is a vessel"""

    def __init__(self, depth):
        self.clf = tree.DecisionTreeClassifier(max_depth=depth)
        # self.clf = self.clf.fit( INPUT, DECISION)

    def save_clf(self, file):
        # something with pickle
        with open(file, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    def fit(self, decision_x, decision_y):
        self.clf.fit(decision_x, decision_y)

    def predict_pixel_proba(self, decision_x):
        return self.clf.predict_proba(decision_x)

    def predict_pixel(self, decision_x):
        return self.clf.predict(decision_x)


class Picture_decision:
    """Model of decisions for each picture."""

    def __init__(self, input_file, output_file, tile_size=5):
        # self.input = get_image(input_file, asgrey=False)
        # self.output = get_image(output_file, asgrey=False)
        self.picture = Picture(input_file, output_file, tile_size=tile_size)
        self.tile_size = tile_size
        self.decision_X = [[], [], [], [], [], [], []]
        self.decision_Y = [[], [], [], [], [], [], []]

    def set_tile_size(self, size):
        self.tile_size = size

    def set_picture(self, input_file, output_file):
        self.picture = Picture(input_file, output_file, tile_size=self.tile_size)

    def clear_decisions(self):
        self.decision_X = [[], [], [], [], [], [], []]
        self.decision_Y = [[], [], [], [], [], [], []]

    def build_decision_model(self, train=False):
        """ pixel color model = 0
            pixel_average_model = 1
            tile_average_model = 2
            hu_model = 3,4,5,6"""
        # several decision models
        self.build_pixel_decision_models(train)
        self.build_tile_decision_models(train)

    def build_pixel_decision_models(self, train=False):
        self.pixel_color_model(0, train)
        self.pixel_average_model(1, train)

    def build_tile_decision_models(self, train=False):
        self.tile_average_model(2, train)
        self.hu_model(3, train)  # 3 .. 6

    def pixel_color_model(self, index=0, train=True):
        '''Based on the pixel color choose output 0/1'''
        for i, row in enumerate(self.picture.input):
            for j, pixel in enumerate(row):
                self.decision_X[index].append(pixel)
                if train:
                    self.decision_Y[index].append(int(self.picture.output[i][j]))

    def pixel_average_model(self, index=1, train=True):
        average_color_per_row = np.average(self.picture.input, axis=0)
        self.average_color = np.average(average_color_per_row, axis=0)
        # self.decision_X.append([])
        # self.decision_Y.append([])
        for i, row in enumerate(self.picture.input):
            for j, pixel in enumerate(row):
                decision_x = list(merge(pixel, self.average_color))
                self.decision_X[index].append(decision_x)
                if train:
                    self.decision_Y[index].append(int(self.picture.output[i][j]))

    def tile_average_model(self, index=2, train=True):
        average_color_per_row = np.average(self.picture.input, axis=0)
        self.average_color = np.average(average_color_per_row, axis=0)
        tile = self.picture.tile(0)
        while tile:
            average_color_per_row = np.average(tile.input, axis=0)
            average_color_per_tile = np.average(average_color_per_row, axis=0)
            decision_x = list(merge(tile.input_pixel, average_color_per_tile, self.average_color))
            self.decision_X[index].append(decision_x)
            if train:
                self.decision_Y[index].append(int(tile.output))
            tile = self.picture.next_tile()

    def hu_model(self, index, train=True):
        """saves decisions to next 4 indexes!"""
        tile = self.picture.tile(0)
        while tile:
            # print(self.picture.current_tile)
            decision_x = self.get_hu(tile.grey)
            self.decision_X[index].append(decision_x)
            decision_x = self.get_hu(tile.red)
            self.decision_X[index + 1].append(decision_x)
            decision_x = self.get_hu(tile.green)
            self.decision_X[index + 2].append(decision_x)
            decision_x = self.get_hu(tile.blue)
            self.decision_X[index + 3].append(decision_x)
            if train:
                self.decision_Y[index].append(tile.output)
                self.decision_Y[index + 1].append(int(tile.output))
                self.decision_Y[index + 2].append(int(tile.output))
                self.decision_Y[index + 3].append(int(tile.output))

            tile = self.picture.next_tile()
            # print(np.any(np.isnan(self.decision_X[index])))
            # print(np.all(np.isfinite(self.decision_X[index])))
            # # where_are_inf = ~np.isfinite(self.decision_X[index])
            # # where_are_nan = ~np.isnan(self.decision_X[index]
            # self.decision_X[index] = np.nan_to_num(self.decision_X[index])
            # self.decision_X[index + 1] = np.nan_to_num(self.decision_X[index + 1])
            # self.decision_X[index + 2] = np.nan_to_num(self.decision_X[index + 2])
            # self.decision_X[index + 3] = np.nan_to_num(self.decision_X[index + 3])
            # print(np.any(np.isnan(self.decision_X[index])))
            # print(np.all(np.isfinite(self.decision_X[index])))

    def get_hu(self, image):
        hu = cv2.HuMoments(cv2.moments(image)).flatten()
        return hu

    def get_log_hu(self, image):
        hu = self.get_hu(image)
        logs = [hu > 0]
        hu[logs] = -np.sign(hu[logs]) * np.log10(np.abs(hu[logs]))
        log_hu = -np.sign(hu) * np.log10(np.abs(hu))
        return log_hu


def build_all_core_models():
    for tile_size in range(1, 51, 2):
        decisions = Picture_decision("Train/input/im01.ppm", "Train/output/m_im01.ppm", tile_size=tile_size)
        decisions.build_decision_model(train=True)
        for i in range(3, 19, 2):
            decisions.set_picture("Train/input/im%.2d.ppm" % i, "Train/output/m_im%.2d.ppm" % i)
            print("Train/input/im%.2d.ppm" % i)
            decisions.build_decision_model(train=True)
        for depth in range(3, 16):
            model = Cart(depth)
            for model_ID, (decision_x, decision_y) in enumerate(zip(decisions.decision_X, decisions.decision_Y)):
                model.fit(decision_x, decision_y)
                model.save_clf("CART/model_%.2d_%.2d_%.2d" % (model_ID, depth, tile_size))
                print("saved", "CART/model_%.2d_%.2d_%.2d" % (model_ID, depth, tile_size))
    print("finish")


if __name__ == "__main__":
    # decisions = Picture_decision("Test/input4x4.ppm", "Test/mask4x4.ppm", tile_size=3)
    # decisions = Picture_decision("Train/input/im01.ppm", "Train/output/m_im01.ppm", tile_size=3)
    # decisions.build_decision_model(train=True)
    # model = Cart(3)
    # for model_ID, (decision_x, decision_y) in enumerate(zip(decisions.decision_X, decisions.decision_Y)):
    #     # print(decision_x)
    #     # print(decision_y)
    #     print(model_ID)
    #     model.fit(decision_x, decision_y)
    #     model.save_clf("CART/model_%.2d" % model_ID)
    # print("finish?")
    build_all_core_models()
    # x = [[0, 1, 0], [0, 0, 1], [5, 0, 0]]
    # x = np.zeros((2,2), dtype=np.double)
    # x[0, 0] = 1
    # x[0, 1] = 10
    # x[1, 0] = 10
    # x[1, 1] = 1
    # print(cv2.HuMoments(cv2.moments(x)).flatten())
    # hu = cv2.HuMoments(cv2.moments(x)).flatten()
    # logs = [hu > 0]
    # hu[logs] = -np.sign(hu[logs]) * np.log10(np.abs(hu[logs]))
    # print(hu)
