"""
Team-name:    changed later
Team-motto:   Memelords know better
Team-members: Florian Eder
              Moritz Enderle
              John Tran


Exemplary predictive model.

You must provide at least 2 methods:
- __init__: Initialization of the class instance
- predict: Uses the model to perform predictions.

The following convenience methods are provided:
- load_microwave_volume: Load three-dimensional microwave image
- visualize_microwave_volume: Visualize the slices of a three-dimensional microwave image
"""

import os
import json
import glob
import copy
import numpy as np
import skimage.io
from matplotlib import pyplot as plt
from cv2 import cvtColor
from detect_single import Detector


def _gray_img(file_path):
    img_array = skimage.io.imread(f'{file_path}')
    img_gray = cvtColor(img_array * 50, 6).astype('uint8')
    img_array = cvtColor(img_gray, 8)

    return img_array


class Model:
    """
    Model to predict, Elon Musk would approve!
    """

    def __init__(self):
        """
        Initialize the class instance

        Important: If you want to refer to relative paths, e.g., './subdir', use
        os.path.join(os.path.dirname(__file__), 'subdir')
        """
        self.model = Detector(device='0')

    def predict(self, data_set_directory):
        """
        This function should provide predictions of labels on a data set.

        Make sure that the predictions are in the correct format for the scoring metric. The method should return an
        array of dictionaries, where the number of dictionaries must match the number of tiff files in
        data_set_dictionary.
        """

        input_files = glob.glob(os.path.join(os.path.abspath(data_set_directory), '*.tiff'))

        example_prediction = {
            'file': '',  # filename of the input image
            'missing_pills': 0,  # number of missing pills
            'present_pills': 0,  # number of present pills
            'coordinates': {
                'missing': [  # centroids of missing pills. The order of the centroids is arbitrary.
                ],
                'present': [  # centroids of present pills. The order of the centroids is arbitrary.
                ]
            }
        }

        file_predictions = []
        for file_path in input_files:
            img = _gray_img(file_path)

            prediction = copy.deepcopy(example_prediction)
            prediction['file'] = os.path.basename(file_path)
            predictions = self.model.detect(img)
            for cls, x, y, _, _, _ in predictions:
                if cls == 1:
                    prediction['present_pills'] += 1
                    prediction['coordinates']['present'].append((x, y))
                else:
                    prediction['missing_pills'] += 1
                    prediction['coordinates']['missing'].append((x, y))
            file_predictions.append(prediction)
        # return list of dictionaries whose length matches the number of tiff files
        return file_predictions

    @staticmethod
    def load_microwave_volume(input_file):
        """
        Load microwave volume from tiff file. Each provided volume contains three slices in propagation direction. The
        provided microwave volumes are given in linear scale.

        :param string input_file: Path to tiff file
        :return ndarray: Image as ndarray with shape MxNx3
        """

        return skimage.io.imread(input_file)

    @staticmethod
    def visualize_microwave_volume(input_file, dynamic_range=25, label=None):
        """
        Visualize the slices of a microwave image volume in logarithmic scale with the given dynamic_range
        :param string input_file: Path to input file
        :param float dynamic_range: Dynamic range in dB (default: 25)
        """

        img = Model.load_microwave_volume(input_file)

        if label is None:
            label_filename = input_file.replace('.tiff', '.json')
            if os.path.exists(label_filename):
                with open(label_filename, 'r', encoding='utf-8') as file:
                    label = json.loads(file.read())

        fig, axs = plt.subplots(1, 3, figsize=(16, 7))
        for i in range(img.shape[2]):
            volume = 20 * np.log10(img[:, :, i])
            max_val = np.max(volume)
            axs[i].imshow(volume, vmax=max_val, vmin=max_val - dynamic_range)
            axs[i].set_title(f"Slice {i + 1:d}")

            if label is not None:
                x_coords = [coord[0] for coord in label['coordinates']['present']]
                y_coords = [257 - coord[1] for coord in label['coordinates']['present']]
                axs[i].scatter(x_coords, y_coords, color='white')

                x_coords = [coord[0] for coord in label['coordinates']['missing']]
                y_coords = [257 - coord[1] for coord in label['coordinates']['missing']]
                axs[i].scatter(x_coords, y_coords, color='red')

        if label is not None:
            fig.canvas.set_window_title(f"Present pills: {len(label['coordinates']['present'])}, "
                                        f"Missing pills: {len(label['coordinates']['missing'])}")

        plt.show()
