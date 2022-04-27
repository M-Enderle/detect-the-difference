#!/usr/bin/env python

"""
Scoring program for the Rohde & Schwarz Engineering Competition 2022

This code computes several metrics based on the predictions exported by the ingestion program.
"""

import os
from sys import argv
import json
import yaml
import tools.metrics
import tools.helpers

# Default I/O directories:
ROOT_DIR = ".."
DEFAULT_INPUT_DIR = os.path.join(ROOT_DIR, "evaluation_results")
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "scoring_output")

if __name__ == "__main__":
    print('\nScoring program started')

    if len(argv) == 1:  # Use the default input and output directories if no arguments are provided
        INPUT_DIR = DEFAULT_INPUT_DIR
        OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    else:
        INPUT_DIR = argv[1]
        OUTPUT_DIR = argv[2]

    # Create the output directory, if it does not already exist and open output files
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    files = []
    for (dirpath, dirnames, filenames) in os.walk(INPUT_DIR):
        files.extend(map(lambda n: os.path.join(*n), zip([dirpath] * len(filenames), filenames)))

    score_file_path = os.path.join(OUTPUT_DIR, 'scores.txt')
    ingestion_score_file_path = os.path.join(INPUT_DIR, 'res', 'ingestion_metrics.json')
    html_file_path = os.path.join(OUTPUT_DIR, 'scores.html')

    # Get all the solution files from the solution directory
    solution_names = tools.helpers.list_files(os.path.join(INPUT_DIR, 'ref_cust', '*.json'))

    if len(solution_names) == 0:
        raise FileNotFoundError("No solution files found.")

    score = {
        'anomaly_detection_accuracy': {'val': 0, 'weighting': 0.45, 'name': 'Anomaly detection accuracy'},
        'avg_sample_accuracy': {'val': 0, 'weighting': 0.25, 'name': 'Average sample accuracy'},
        'pylint': {'val': 0, 'weighting': 0.1, 'name': 'Code quality'},
        'distance': {'val': 0, 'weighting': 0.05, 'name': 'Distance score'},
        'prediction_time': {'val': 0, 'weighting': 0, 'name': 'Prediction time'},
        'elapsed_time': {'val': 0, 'weighting': 0, 'name': 'Total elapsed time'},
        'total': {'val': 0, 'weighting': 0, 'name': 'Total score'}
    }

    # Read the execution time and add it to the scores:
    if os.path.exists(os.path.join(INPUT_DIR, 'res', 'metadata')):
        try:
            with open(os.path.join(INPUT_DIR, 'res', 'metadata'), 'r') as metadata_file:
                metadata = yaml.load(metadata_file, Loader=yaml.SafeLoader)
                score['elapsed_time']['val'] = metadata['elapsedTime']
        except Exception as e:
            print('Error:', e)

    solutions = []
    predictions = []

    for i, solution_file in enumerate(solution_names):
        # Get the last prediction from the res subdirectory (must end with '.predict')
        predict_file = os.path.join(
            INPUT_DIR, 'res', f"{os.path.splitext(os.path.basename(solution_file))[0]}.prediction")
        if not os.path.exists(predict_file):
            raise IOError('Missing prediction file {}'.format(predict_file))
        # Read the solution and prediction values into numpy arrays
        solutions.append(tools.helpers.load_results(solution_file))
        predictions.append(tools.helpers.load_results(predict_file))

    print(f"Scoring is based on {len(predictions):d} samples.")

    # Compute scoring metrics
    score['anomaly_detection_accuracy']['val'] = tools.metrics.anomaly_detection_accuracy(solutions, predictions)
    score['avg_sample_accuracy']['val'] = tools.metrics.avg_sample_accuracy(solutions, predictions)
    score['distance']['val'] = tools.metrics.distance(solutions, predictions)

    if os.path.exists(ingestion_score_file_path):
        with open(ingestion_score_file_path, 'r') as ingestion_score_file:
            ingestion_scores = json.load(ingestion_score_file)
            score['pylint']['val'] = ingestion_scores['pylint_rating']
            score['prediction_time']['val'] = ingestion_scores['prediction_time']
    else:
        print(f"File {os.path.basename(ingestion_score_file_path)} does not exist.")

    # Compute total score
    score['total']['val'] = sum([score_dict['val'] * score_dict['weighting'] for score_dict in score.values()])

    with open(score_file_path, 'w') as score_file:
        for score_key, score_value in score.items():
            score_file.write(f"{score_key}: {score_value['val']:.6f}\n")
            print(f"{score_value['name']:26}: {score_value['val']:>10.6f}")

    print('Scoring program completed.')
