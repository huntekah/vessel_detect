from skimage import io
import numpy as np


class Picture:
    def __init__(self, path_input, path_output=None, tile_size=5):
        self.load_image(path_input, path_output)
        self.shape = self.input.shape
        self.current_tile = 0
        self.tile_size = tile_size

    def tile(self, n=None):
        """ n represents nth tile of size tile_zize square """
        if isinstance(n, int):
            self.current_tile = n
        else:
            n = self.current_tile
        y = int(n % (self.shape[1] - self.tile_size + 1))
        x = int(np.floor(n / (self.shape[1] - self.tile_size + 1)))
        x = x % (self.shape[0] - self.tile_size + 1)

        class Tile:
            class Centre:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y

                def shit(self):
                    pass

            def __init__(self):
                pass

            def set_input(self, tile):
                self.input = tile
                len = tile.shape[0]
                self.input_pixel = tile[int((len - 1) / 2)][int((len - 1) / 2)]

            def set_red(self, tile):
                self.red = tile

            def set_green(self, tile):
                self.green = tile

            def set_blue(self, tile):
                self.blue = tile

            def set_grey(self, tile):
                self.grey = tile

            def set_output(self, pixel):
                self.output = pixel

            def set_centre(self, x, y):
                self.centre = self.Centre(x=x, y=y)

            def __bool__(self):
                return True

        result = Tile()
        result.set_input(self.input[x:(x + self.tile_size), y:(y + self.tile_size), :])
        result.set_red(self.red[x:x + self.tile_size, y:y + self.tile_size])
        result.set_green(self.green[x:x + self.tile_size, y:y + self.tile_size])
        result.set_blue(self.blue[x:x + self.tile_size, y:y + self.tile_size])
        result.set_grey(self.grey[x:x + self.tile_size, y:y + self.tile_size])
        if hasattr(self, 'output'):
            result.set_output(self.output[x + int((self.tile_size - 1) / 2), y + int((self.tile_size - 1) / 2)])
            # print(x + int((self.tile_size - 1) / 2), y + int((self.tile_size - 1) / 2))
        result.set_centre(x + int((self.tile_size - 1) / 2), y + int((self.tile_size - 1) / 2))
        return result

    def next_tile(self):
        pass
        if self.current_tile >= (self.shape[0] - self.tile_size + 1) * (self.shape[1] - self.tile_size + 1) - 1:
            return False
        result = self.tile()
        self.current_tile += 1
        return result

    def load_image(self, path_input, path_output=None):
        self.input = self.get_image(path_input, asgrey=False)
        self.red = self.input[:, :, 0]
        self.green = self.input[:, :, 1]
        self.blue = self.input[:, :, 2]
        self.grey = self.get_image(path_input, asgrey=True, _flatten=True)
        if path_output:
            self.output = self.get_image(path_output, asgrey=True, _flatten=True)

    def get_image(self, path, asgrey=True, _flatten=False):
        return io.imread(path, as_grey=asgrey, flatten=_flatten)


if __name__ == "__main__":
    # image = picture("Test/input/im21.ppm", "Test/output/m_im21.ppm")
    image = Picture("Test/input3x3.ppm", "Test/mask3x3.ppm", tile_size=3)
    tile = image.tile()
    i = 0
    while tile:
        print(i, tile.output)
        i += 1
        tile = image.next_tile()
        # print("input\n")
        # print(tile.input)
        # print("red\n")
        # print(tile.red)
        # print("green\n")
        # print(tile.green)
        # print("blue\n")
        # print(tile.blue)
        # print("grey\n")
        # print(tile.grey)
        # print("output\n")
        # print(tile.output)
