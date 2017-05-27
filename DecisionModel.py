from sklearn import tree
import numpy as np
import pickle
from skimage import io
from vessel_detect.picture import Picture


def get_image(path, asgrey=True, _flatten=False):
    return io.imread(path, as_grey=asgrey, flatten=_flatten)


class Cart:
    '''we use it to create model, that will predict if a pixel is a vessel'''

    def __init__(self, depth):
        self.clf = tree.DecisionTreeClassifier(max_depth=depth)
        # self.clf = self.clf.fit( INPUT, DECISION)

    def save_clf(self, file):
        # something with pickle
        pickle.dump(self.clf, file)


class Picture_decision:
    '''Usedto create a decision model'''

    def __init__(self, input_file, output_file, tile_size=5):
        # self.input = get_image(input_file, asgrey=False)
        # self.output = get_image(output_file, asgrey=False)
        self.picture = Picture(input_file, output_file, tile_size=tile_size)
        self.tile_size = tile_size
        self.decision_X = [[], [], [], [], [], [], []]
        self.decision_Y = [[], [], [], [], [], [], []]

    def set_tile_size(self, size):
        self.tile_size = size

    def clear_decisions(self):
        self.decision_X = [[], [], [], [], [], [], []]
        self.decision_Y = [[], [], [], [], [], [], []]

    def build_decision_model(self):
        # several decision models
        self.pixel_color_model(0)
        self.pixel_average_model(1)
        self.tile_average_model(2)

    def set_picture(self, input_file, output_file):
        self.picture = Picture(input_file, output_file, tile_size=self.tile_size)

    def pixel_color_model(self, index=0):
        '''Based on the pixel color choose output 0/1'''
        for i, row in enumerate(self.picture.input):
            for j, pixel in enumerate(row):
                self.decision_X[index].append(pixel)
                # one output not multiple!
                self.decision_Y[index].append(self.picture.output[i][j])

    def pixel_average_model(self, index=1):
        average_color_per_row = np.average(self.picture.input, axis=0)
        self.average_color = np.average(average_color_per_row, axis=0)
        # self.decision_X.append([])
        # self.decision_Y.append([])
        for i, row in enumerate(self.picture.input):
            for j, pixel in enumerate(row):
                decision_x = [pixel, self.average_color]
                self.decision_X[index].append(decision_x)
                self.decision_Y[index].append(self.picture.output[i][j])

    def tile_average_model(self, index=2):
        average_color_per_row = np.average(self.picture.input, axis=0)
        self.average_color = np.average(average_color_per_row, axis=0)
        tile = self.picture.next_tile()
        while tile:
            average_color_per_row = np.average(tile.input, axis=0)
            average_color_per_tile = np.average(average_color_per_row, axis=0)
            decision_x = [tile.input_pixel, average_color_per_tile, self.average_color]
            self.decision_X[index].append(decision_x)
            self.decision_Y[index].append(tile.output)
            tile = self.picture.next_tile()


if __name__ == "__main__":
    # decisions = Picture_decision("Test/input5x5.ppm", "Test/mask5x5.ppm", tile_size=3)
    # decisions.tile_average_model()
    # print(decisions.decision_X)
    # print("\n")
    # print(decisions.decision_Y)
    a = []
    a.extend([[], [], [], [], [], []])
    b = []
    b.append([])
    b.append([])
    b.append([])
    c = [[], [], [], [], [], []]
    print(a)
    print(b)
    print(c)
