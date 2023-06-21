import pattern
from pattern import Checker
from pattern import Circle
from pattern import Spectrum
import generator
from generator import ImageGenerator
import NumpyTests
from NumpyTests import TestCheckers
from NumpyTests import TestCircle
from NumpyTests import TestSpectrum
from NumpyTests import TestGen
import unittest

#Checker pattern
checker = Checker(100, 10)  #resolution, tile size
checker.draw()
checker.show()

#Circle pattern
circle = Circle(1000, 200, (400, 600))   #resolution, radius, position
circle.draw()
circle.show()

#Spectrum pattern
spectrum = Spectrum(2500)  # resolution
spectrum.draw()
spectrum.show()

#Generator
file_path = 'exercise_data'
label_path = 'Labels.json'
batch_size= 10
image_size = [32, 32, 5]

gen=ImageGenerator(file_path, label_path, batch_size, [32, 32, 3], False, False, True)
gen.next()
gen.show()
if __name__ == '__main__':
    unittest.main()