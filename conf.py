""" cv-tbox Split Creator - Configuration File """

import os


# Locations

CV_COMPRESSED_BASE_DIR: str = os.path.join("V:", os.sep, "DATASETS", "VOICE", "CV")
CV_DATASET_BASE_DIR: str = os.path.join("M:", os.sep, "DATASETS", "CV")

# Version to work on

CV_DATASET_VERSION: str = "cv-corpus-17.0-2024-03-15"

# Splitting Parameters

# At this point statistical sampling give better results
SAMPLE_SIZE_THRESHOLD: int = 150000
# We do not deal with languages having this much records in validated, for vw and vx algorithmns
MIN_VALIDATED_THRESHOLD = 2000

# Program parameters

# If true, report all on different lines, else show only generated
VERBOSE: bool = False
# If true, fail if source is not found, else skip it
FAIL_ON_NOT_FOUND: bool = True
# If true, regenerate the splits even if they exist,
FORCE_CREATE: bool = False
