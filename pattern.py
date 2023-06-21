import numpy as np
import matplotlib.pyplot as plt
import copy

class Checker:
    def __init__(self, res, tsize):
        self.res = res
        self.t_size = tsize
        self.output = np.zeros((res, res))

    def draw(self):
        if self.res % (2 * self.t_size) != 0:
            print("Cannot draw the checkerboard!")

        else:
            blk = np.zeros((self.t_size, self.t_size))
            wht = np.ones((self.t_size, self.t_size))
            merge = np.concatenate((blk, wht), axis=1)
            merge = np.concatenate((merge, np.flip(merge, axis=1)),axis=0)
            rep = int((self.res / self.t_size) / 2)
            self.output = np.tile(merge, (rep, rep))

        return self.output.copy()

    def show(self):
        plt.imshow(self.draw(), cmap='gray')
        plt.axis('off')
        plt.show()


class Circle:
    def __init__(self, resolution, radius, centers):
        self.resolution = resolution
        self.radius = radius
        self.centers = tuple(centers)
        self.output = np.zeros((resolution, resolution))
    def draw(self):
        x_axis = np.arange(self.resolution)
        y_axis = np.arange(self.resolution)
        xx, yy = np.meshgrid(x_axis, y_axis)
        x_centers, y_centers = self.centers
        self.output = (((xx - x_centers) ** 2) + ((yy - y_centers) ** 2) <= self.radius ** 2)
        return self.output.copy()

    def show(self):
        plt.imshow(self.draw(), cmap='gray')
        plt.axis('off')
        plt.show()


class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = self.draw()

    def draw(self):
        resolution = self.resolution
        img = np.zeros([resolution, resolution, 3])
        img[:, :, 0] = np.linspace(0, 1, resolution)
        img[:, :, 1] = np.linspace(0, 1, resolution).reshape(resolution, 1)
        img[:, :, 2] = np.linspace(1, 0, resolution)

        out = copy.copy(img)
        return out

    def show(self):
        plt.imshow(self.output)
        plt.show()