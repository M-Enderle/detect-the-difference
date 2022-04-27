#!/usr/bin/env python

""" Implementation of the ingestion program """

# Usage: python ingestion.py input_dir output_dir hidden_dir ingestion_program_dir submission_program_dir

# AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
#
# This is the "ingestion program" written by the organizers.
# This program also runs on the challenge platform to test your code.
#
# Main contributors: Marius Brinkmann
# Last modifications Marius Brinkmann, March 2022

# =============================================================================
# =========================== BEGIN OPTIONS ===================================
# =============================================================================

# Recommended to keep VERBOSE = True: shows various progression messages
VERBOSE = True  # output messages to stdout and stderr for debug purposes

# I/O defaults
# Use default location for the input and output data:
# If no arguments to run.py are provided, this is where the data will be found
# and the results written to. Change the root_dir to your local directory.
ROOT_DIR = "../"
DEFAULT_INPUT_DIR = ROOT_DIR + "public_data"
DEFAULT_OUTPUT_DIR = ROOT_DIR + "evaluation_results"
DEFAULT_HIDDEN_DIR = r'E:\measurements\engineering_competition\dataset_final\doppelherzaz\augmentation-0' # ROOT_DIR + "validation_data"
DEFAULT_PROGRAM_DIR = ROOT_DIR + "ingestion_program"
DEFAULT_SUBMISSION_DIR = ROOT_DIR + "test-submission"

# =============================================================================
# =========================== END USER OPTIONS ================================
# =============================================================================


if __name__ == "__main__":
    import os
    from sys import argv, path
    import json
    import timeit
    import tensorflow.config
    import tools.helpers

    print('Ingestion program started.')

    # INPUT/OUTPUT: Get input and output directory names
    if len(argv) == 1:  # Use the default input and output directories if no arguments are provided
        INPUT_DIR = DEFAULT_INPUT_DIR
        OUTPUT_DIR = DEFAULT_OUTPUT_DIR
        HIDDEN_DIR = DEFAULT_HIDDEN_DIR
        PROGRAM_DIR = DEFAULT_PROGRAM_DIR
        SUBMISSION_DIR = DEFAULT_SUBMISSION_DIR
    else:
        INPUT_DIR = os.path.abspath(argv[1])
        OUTPUT_DIR = os.path.abspath(argv[2])
        HIDDEN_DIR = os.path.abspath(argv[3])
        PROGRAM_DIR = os.path.abspath(argv[4])
        SUBMISSION_DIR = os.path.abspath(argv[5])

    if VERBOSE:
        print(f"Using input_dir: {INPUT_DIR}")
        print(f"Using output dir: {OUTPUT_DIR}")
        print(f"Using program_dir: {PROGRAM_DIR}")
        print(f"Using submission_dir: {SUBMISSION_DIR}")

        print(f"Available GPU(s): {tensorflow.config.list_physical_devices('GPU')}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Our libraries
    path.append(PROGRAM_DIR)
    path.append(SUBMISSION_DIR)

    metrics = dict()
    metrics['pylint_rating'] = tools.helpers.run_pylint(os.path.join(SUBMISSION_DIR, 'model.py'), verbose=True)
    metrics['prediction_time'] = 0

    from model import Model  # example model

    # Create a model
    print("Creating model instance")
    M = Model()

    prediction_time = 0

    # Iterate through data set(s)
    dataset_dirs = [HIDDEN_DIR]  # INPUT_DIR,
    for dataset_dir in dataset_dirs:
        tic = timeit.default_timer()
        predictions = M.predict(dataset_dir)
        metrics['prediction_time'] += timeit.default_timer() - tic

        for prediction in predictions:
            prediction_file = os.path.join(OUTPUT_DIR, f"{os.path.splitext(prediction['file'])[0]}.prediction")
            # print(f"Writing file {os.path.relpath(prediction_file)}")
            with open(prediction_file, 'w') as prediction_file:
                prediction_file.write(json.dumps(prediction))

    print(f"Performed {len(predictions):d} predictions in {metrics['prediction_time']:.3g} s")

    with open(os.path.join(OUTPUT_DIR, 'ingestion_metrics.json'), 'w') as metric_file:
        metric_file.write(json.dumps(metrics))

    print('Ingestion program completed.')
