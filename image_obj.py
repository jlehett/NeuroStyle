import cv2
import numpy as np

class Image:
    def __init__(self, subsection_radius):
        self.image = None
        self.radius = subsection_radius

    def openImage(self, filename):
        self.image = cv2.imread(filename)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = self.image.astype(dtype=np.float32)
        self.image /= 255.0

    def setImage(self, image):
        self.image = image

    def getPixelSection(self, x, y):
        size = self.radius * 2 + 1
        new_array = np.zeros((size, size))
        startx, starty = x-self.radius, y-self.radius
        endx, endy = x+self.radius+1, y+self.radius+1
        centerx, centery = self.radius, self.radius
        array_x, array_y = 0, 0
        end_array_x, end_array_y = size, size
        while startx < 0:
            array_x += 1
            startx += 1
        while endx >= self.image.shape[1]:
            end_array_x -= 1
            endx -= 1
        while starty < 0:
            array_y += 1
            starty += 1
        while endy >= self.image.shape[0]:
            end_array_y -= 1
            endy -= 1
        new_array[array_y:end_array_y, array_x:end_array_x] = self.image[starty:endy, startx:endx]
        return new_array

    def getPixel(self, x, y):
        return self.image[y][x]