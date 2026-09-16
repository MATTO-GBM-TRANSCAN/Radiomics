#!/usr/bin/env python

from __future__ import print_function

import argparse
import csv
import logging
import os
import shutil
import threading
from collections import OrderedDict
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import SimpleITK as sitk

import radiomics
import radiomics.imageoperations as imageoperations
from radiomics.featureextractor import RadiomicsFeatureExtractor

threading.current_thread().name = "Main"

# Command-line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--input_csv",
    default="/input/input.csv",
    help="Path (inside the container) to the input CSV listing Image/Mask pairs.",
)
args = argparser.parse_args()

# File variables
ROOT = os.getcwd()
PARAMS = "/input/extraction_parameters.yaml"  # Parameter file, provided by the user alongside input.csv
LOG = f"/output/log_{datetime.now().strftime('%Y-%m-%d')}.txt"  # Location of output log file
INPUTCSV = args.input_csv
OUTPUTCSV = "/output/discretization.csv"

# Parallel processing variables
TEMP_DIR = "/output/temp"
REMOVE_TEMP_DIR = True  # Remove temporary directory when results have been successfully stored into 1 file
NUM_OF_WORKERS = 32
HEADERS = None  # headers of all extracted features

# Assumes the input CSV has at least 2 columns: "Image" and "Mask"
# These columns indicate the location of the image file and mask file, respectively
# Additionally, this script uses 2 additonal Columns: "Patient" and "Reader"
# These columns indicate the name of the patient (i.e. the image), the reader (i.e. the segmentation), if
# these columns are omitted, a value is automatically generated ("Patient" = "Pt <Pt_index>", "Reader" = "N/A")

# Assumes the mounted /input folder contains both input.csv (input csv file)
# and extraction_parameters.yaml (settings)
# Creates a log file in the output folder

# Set up logging
################

rLogger = radiomics.logger
# rLogger.setLevel(logging.DEBUG)
logHandler = logging.FileHandler(filename=LOG, mode="a")
logHandler.setLevel(logging.DEBUG)
logHandler.setFormatter(
    logging.Formatter("%(levelname)-.1s: (%(threadName)s) %(name)s: %(message)s")
)
rLogger.addHandler(logHandler)


# Define filter that allows messages from specified filter and level INFO and up, and level WARNING and up from other
# loggers.
class info_filter(logging.Filter):
    def __init__(self, name):
        super(info_filter, self).__init__(name)
        self.level = logging.WARNING

    def filter(self, record):
        if record.levelno >= self.level:
            return True
        if record.name == self.name and record.levelno >= logging.INFO:
            return True
        return False


# Adding the filter to the first handler of the radiomics logger limits the info messages on the output to just those
# from radiomics.batch, but warnings and errors from the entire library are also printed to the output. This does not
# affect the amount of logging stored in the log file.
outputhandler = rLogger.handlers[0]  # Handler printing to the output
outputhandler.setFormatter(
    logging.Formatter("[%(asctime)-.19s] (%(threadName)s) %(name)s: %(message)s")
)
outputhandler.setLevel(
    logging.INFO
)  # Ensures that INFO messages are being passed to the filter
outputhandler.addFilter(info_filter("radiomics.batch"))

logging.getLogger("radiomics.batch").debug("Logging init")


def run(case):
    global PARAMS, ROOT, TEMP_DIR
    ptLogger = logging.getLogger("radiomics.batch")

    result_vector = OrderedDict(case)

    try:
        # set thread name to patient name
        threading.current_thread().name = case["Patient"]

        filename = (
            r"discretization_"
            + str(case["Reader"])
            + "_"
            + str(case["Patient"])
            + ".csv"
        )
        output_filename = os.path.join(ROOT, TEMP_DIR, filename)

        t = datetime.now()

        imageFilepath = case["Image"]  # Required
        maskFilepath = case["Mask"]  # Required
        label = case.get("Label", None)  # Optional

        settings = {
            "normalize": True,  # Whether to apply image intensity normalization
            "normalizeScale": 100,  # Rescale intensities to a certain range (used in z-score normalization)
            "voxelArrayShift": "300",
            "interpolator": "sitkBSpline",
            "resampledPixelSpacing": [
                0.5,
                0.5,
                1,
            ],  # If None, no resampling is performed
        }

        image, mask = RadiomicsFeatureExtractor.loadImage(
            imageFilepath, maskFilepath, **settings
        )
        normalized_image = imageoperations.normalizeImage(image, **settings)
        matrix = sitk.GetArrayFromImage(normalized_image)
        mask = sitk.GetArrayFromImage(mask) == 1
        result_vector["ROI_min"] = np.min(matrix[mask])
        result_vector["ROI_max"] = np.max(matrix[mask])

        for bin_width in [5, 6, 7, 8, 9, 10, 11, 12]:
            settings.update({"binWidth": bin_width})
            matrix, _ = imageoperations.binImage(matrix, mask, **settings)
            # result_vector["gray_levels"] = np.unique(matrix[mask])
            result_vector[f"n_bins_{bin_width}"] = np.unique(matrix[mask]).shape[0]
            matrix = sitk.GetArrayFromImage(normalized_image)

        # Store results in temporary separate files to prevent write conflicts
        # This allows for the extraction to be interrupted. Upon restarting, already processed cases are found in the
        # TEMP_DIR directory and loaded instead of re-extracted
        with open(output_filename, "w") as outputFile:
            writer = csv.DictWriter(
                outputFile,
                fieldnames=list(result_vector.keys()),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerow(result_vector)

        # Display message

        delta_t = datetime.now() - t

        ptLogger.info(
            "Patient %s read by %s processed in %s",
            case["Patient"],
            case["Reader"],
            delta_t,
        )

    except Exception:
        ptLogger.error("Feature extraction failed!", exc_info=True)

    return result_vector


def _writeResults(featureVector):
    global HEADERS, OUTPUTCSV

    # Use the lock to prevent write access conflicts
    try:
        with open(OUTPUTCSV, "a") as outputFile:
            writer = csv.writer(outputFile, lineterminator="\n")
            if HEADERS is None:
                HEADERS = list(featureVector.keys())
                writer.writerow(HEADERS)

            row = []
            for h in HEADERS:
                row.append(featureVector.get(h, "N/A"))
            writer.writerow(row)
    except Exception:
        logging.getLogger("radiomics.batch").error(
            "Error writing the results!", exc_info=True
        )


if __name__ == "__main__":
    logger = logging.getLogger("radiomics.batch")

    # Ensure the entire extraction is handled on 1 thread
    #####################################################

    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)

    # Set up the pool processing
    ############################

    logger.info("pyradiomics version: %s", radiomics.__version__)
    logger.info("Loading CSV...")

    # Extract List of cases
    cases = []
    try:
        with open(INPUTCSV, "r") as inFile:
            cr = csv.DictReader(inFile, lineterminator="\n")
            cases = []
            for row_idx, row in enumerate(cr, start=1):
                # If not included, add a "Patient" and "Reader" column.
                if "Patient" not in row:
                    row["Patient"] = row_idx
                if "Reader" not in row:
                    row["Reader"] = "N-A"
                cases.append(row)

    except Exception:
        logger.error("CSV READ FAILED", exc_info=True)

    logger.info("Loaded %d jobs", len(cases))

    # Make output directory if necessary
    if not os.path.isdir(os.path.join(ROOT, TEMP_DIR)):
        logger.info(
            "Creating temporary output directory %s", os.path.join(ROOT, TEMP_DIR)
        )
        os.mkdir(os.path.join(ROOT, TEMP_DIR))

    # Start parallel processing
    ###########################

    logger.info(
        "Starting parralel pool with %d workers out of %d CPUs",
        NUM_OF_WORKERS,
        cpu_count(),
    )
    # Running the Pool
    pool = Pool(NUM_OF_WORKERS)
    results = pool.map(run, cases)

    try:
        # Store all results into 1 file
        with open(OUTPUTCSV, mode="w") as outputFile:
            writer = csv.DictWriter(
                outputFile,
                fieldnames=list(results[0].keys()),
                restval="",
                extrasaction="raise",  # raise error when a case contains more headers than first case
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(results)

        if REMOVE_TEMP_DIR:
            logger.info(
                "Removing temporary directory %s (contains individual case results files)",
                os.path.join(ROOT, TEMP_DIR),
            )
            shutil.rmtree(os.path.join(ROOT, TEMP_DIR))
    except Exception:
        logger.error("Error storing results into single file!", exc_info=True)
