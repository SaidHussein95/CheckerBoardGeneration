# Pattern Generation and Data Handling

This repository contains Python classes and scripts for generating patterns and handling data for machine learning applications. 
The repository is organized into two main sections: pattern generation and data handling.

## Pattern Generation

The `pattern.py` file includes classes for generating various patterns using numpy functions. The available patterns are:

- Checkerboard: A checkerboard pattern with customizable tile size and resolution.
- Circle: A binary circle pattern with a given radius and position.
- Color Spectrum: An RGB color spectrum pattern.

Each pattern is implemented as a separate Python class, providing methods for initialization, drawing the pattern, and visualization. The patterns are stored as numpy arrays.

To visualize the patterns, the `main.py` script imports and calls the pattern classes. This script can also be used for debugging purposes.

## Data Handling

The `generator.py` file contains the `ImageGenerator` class, which facilitates the reading and processing of image data for machine learning tasks. The class is designed to handle image files and their associated class labels stored in a JSON file.

The `ImageGenerator` class provides the following functionalities:

- Reading a directory of image files and a JSON file with labels.
- Generating batches of images and labels for training.
- Resizing images to a desired size.
- Optional data augmentation techniques: shuffling, mirroring, and rotation.
- Access to the current epoch number and class name mappings.

The `main.py` script demonstrates the usage of the `ImageGenerator` class and includes a `show()` method for visualizing batches of images with their corresponding labels.

## Install the required dependencies:
pip install numpy matplotlib scikit-image

## Usage:
Generate and visualize patterns: python main.py

## Contributing
Contributions to this repository are welcome. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.
