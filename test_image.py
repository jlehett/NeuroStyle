import numpy as np
import cv2
import sys

class TestImage:
    def __init__(self, size=None, subsection_radius=4):
        self.image = None
        if size == None:
            self.size = None
        else:
            self.size = size.get()
        self.radius = subsection_radius
        if subsection_radius != 4:
            self.radius = subsection_radius.get()

    def setImage(self, image):
        self.image = image

    def openImageTraining(self, filename):
        self.image = cv2.imread(filename)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = self.image.astype(dtype=np.float32)
        self.image /= 255.0

    def openImage(self, filename):
        self.image = cv2.imread(filename)
        self.image = cv2.resize(self.image, (self.size, self.size))
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = self.image.astype(dtype=np.float32)
        self.image /= 255.0

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
        while endx >= self.image.shape[0]:
            end_array_x -= 1
            endx -= 1
        while starty < 0:
            array_y += 1
            starty += 1
        while endy >= self.image.shape[1]:
            end_array_y -= 1
            endy -= 1
        try:
            new_array[array_y:end_array_y, array_x:end_array_x] = self.image[starty:endy, startx:endx]
        except:
            new_array[array_y:end_array_y, array_x:end_array_x] = 0
        return new_array

    def getPrediction(self, model):
        new_image = np.zeros((self.image.shape[1], self.image.shape[0]))
        best_percent = 10
        print()
        for y in range(self.image.shape[1]):
            current_percent = y / self.image.shape[1] * 100.0
            if current_percent > best_percent:
                print_percent = 'CURRENT PROGRESS: ' + str(current_percent) + '% ...'
                sys.stdout.write('\r'+print_percent)
                best_percent = current_percent + 10
            for x in range(self.image.shape[0]):
                predict_array = np.array([self.getPixelSection(x, y)])
                new_image[y][x] = model.predict(predict_array)
        return new_image

    def getPixel(self, x, y):
        return self.image[y][x]