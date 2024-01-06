""" cv-tbox Split Creator - Configuration File """
import os


# Locations

CV_COMPRESSED_BASE_DIR: str = os.path.join("V:", os.sep, "DATASETS", "VOICE", "CV")
CV_DATASET_BASE_DIR: str = os.path.join("M:", os.sep, "DATASETS", "CV")

# Version to work on

CV_DATASET_VERSION: str = "cv-corpus-16.1-2023-12-06"

# Program parameters

# If true, report all on different lines, else show only generated
VERBOSE: bool = False
# If true, fail if source is not found, else skip it
FAIL_ON_NOT_FOUND: bool = True
# If true, regenerate the splits even if they exist,
FORCE_CREATE: bool = False
