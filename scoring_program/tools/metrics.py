import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def anomaly_detection_accuracy(labels, predictions):
    """ Sample accuracy metric """

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for (label, prediction) in zip(labels, predictions):
        if label['missing_pills'] > 0 and prediction['missing_pills'] > 0:
            true_positives += 1
        elif label['missing_pills'] == 0 and prediction['missing_pills'] == 0:
            true_negatives += 1
        elif label['missing_pills'] > 0 and prediction['missing_pills'] == 0:
            false_negatives += 1
        elif label['missing_pills'] == 0 and prediction['missing_pills'] > 0:
            false_positives += 1

    sample_accuracy = (true_positives + true_negatives) \
                      / (true_positives + true_negatives + false_positives + false_negatives)

    # precision = true_positives / (true_positives + false_positives)
    # recall = true_positives / (true_positives + false_negatives)
    # f_score = 2 * precision * recall / (precision + recall)

    return 100 * sample_accuracy


def accuracy(labels, predictions):
    """ Accuracy metric """

    accuracy_val = sum([
        1 for (label, prediction) in zip(labels, predictions)
        if label['present_pills'] == prediction['present_pills']
        and label['missing_pills'] == prediction['missing_pills']
    ]) / len(labels)

    return 100 * accuracy_val


def slot_count_accuracy(labels, predictions):
    """ Slot count accuracy """

    number_of_slots = {'labels': 0, 'predictions': 0}
    for (prediction, label) in zip(predictions, labels):
        number_of_slots['labels'] += label['missing_pills'] + label['present_pills']
        number_of_slots['predictions'] += prediction['missing_pills'] + prediction['present_pills']

    slot_count_accuracy_val = 1 - abs(
        number_of_slots['labels'] - number_of_slots['predictions']
    ) / number_of_slots['labels']

    return 100 * slot_count_accuracy_val


def slot_count_accuracy2(labels, predictions):
    """ Slot count accuracy 2 """

    number_of_present_pills = {'labels': 0, 'predictions': 0}
    number_of_missing_pills = {'labels': 0, 'predictions': 0}
    for (prediction, label) in zip(predictions, labels):
        number_of_present_pills['labels'] += label['present_pills']
        number_of_present_pills['predictions'] += prediction['present_pills']

        number_of_missing_pills['labels'] += label['missing_pills']
        number_of_missing_pills['predictions'] += prediction['missing_pills']

    slot_count_accuracy2_val = (number_of_present_pills['predictions'] + number_of_missing_pills['predictions']) / \
                               (number_of_present_pills['labels'] + number_of_missing_pills['labels'])

    return 100 * slot_count_accuracy2_val


def avg_sample_accuracy(labels, predictions):
    """ Average sample accuracy """

    sample_deviation = []
    for (prediction, label) in zip(predictions, labels):
        total_number_of_slots = label['missing_pills'] + label['present_pills']
        wrong_detections = (abs(label['missing_pills'] - prediction['missing_pills'])
                            + abs(label['present_pills'] - prediction['present_pills'])) / total_number_of_slots
        sample_deviation.append(wrong_detections)

    avg_sample_accuracy_val = 1 - np.mean(sample_deviation)

    return 100 * avg_sample_accuracy_val


def distance(labels, predictions):
    """ Compute rating for Euclidean distance """

    def weighting(input_val, mu=0, sigma=10):
        """ Convert Euclidean distance to score using Gaussian weighting """

        return np.exp(-(input_val - mu) ** 2 / (2 * sigma ** 2))

    def gaussian_score(points_ref, points):
        """ Compute score for set of points """

        points_ref = np.array(points_ref)
        points = np.array(points)

        cost_matrix = cdist(points_ref, points)  # Euclidean norm

        row_assignment, col_assignment = linear_sum_assignment(cost_matrix)

        distances = cost_matrix[row_assignment, col_assignment]
        return np.mean(weighting(distances))

    scores = {
        'present': np.zeros(len(labels)),
        'missing': np.zeros(len(labels)),
    }
    number_of_pills = {
        'present': 0,
        'missing': 0
    }
    for label_index, (label, prediction) in enumerate(zip(labels, predictions)):
        for pill_type in ('present', 'missing'):
            if len(label['coordinates'][pill_type]) > 0 and len(prediction['coordinates'][pill_type]) > 0:
                scores[pill_type][label_index] = gaussian_score(label['coordinates'][pill_type],
                                                                prediction['coordinates'][pill_type])
                number_of_pills[pill_type] += len(label['coordinates'][pill_type])

    total_number_of_pills = number_of_pills['present'] + number_of_pills['missing']
    return 100 * (np.mean(scores['present']) * number_of_pills['present']
                  + np.mean(scores['missing']) * number_of_pills['missing']) / total_number_of_pills
