"""cv-tbox Split Creator - Configuration File"""

import os

#
# Locations
#

# Where you keep your compressed datasets (downloaded .tar.gz files)
CV_COMPRESSED_BASE_DIR: str = os.path.join("X:", os.sep, "DS_COMPRESSED", "VOICE", "CV")
# Where you keep your expanded datasets (will be expanded like <CV_DATASET_BASE_DIR>/cv-corpus-19.0-2024-09-13/<lc>)
CV_EXTRACTED_BASE_DIR: str = os.path.join("T:", os.sep, "TBOX", "ds_ext", "cv")
# - If you keep metadata with clips (e.g. working on a single language) set this to be same as CV_DATASET_BASE_DIR
# - If you work with all CV languages, you might want to have them in another place.
# With many versions the metadata itself reaches hundreds of GBs (for all CV languages).
# E.g. you could use some compression on them, "compact" in Windows for example.
# If this is the case, CV_METADATA_BASE_DIR should point to that directory.
CV_METADATA_BASE_DIR: str = os.path.join("T:", os.sep, "TBOX", "ds_ext", "cv")

# Base directory for split-maker (SM) data (experiments and results)
SM_DATA_DIR: str = os.path.join("T:", os.sep, "GITREPO_DATA", "cv-tbox-split-maker")

# - Latest FULL Version to work on
# - Latest Delta Version to work merge (if you are using delta upgrade)
CV_FULL_VERSION: str = "cv-corpus-22.0-2025-06-20"
CV_DELTA_VERSION: str = "cv-corpus-22.0-delta-2025-06-20"

# Previous FULL Version to upgrade with delta (if you are using delta upgrade)
CV_FULL_PREV_VERSION: str = "cv-corpus-21.0-2025-03-14"

#
# Splitting Parameters
#

# At this point statistical sampling give better results (valid for v1, vw, vx algorithms)
SAMPLE_SIZE_THRESHOLD: int = 150000
# We do not deal with languages having less than this much records in validated, for vw and vx algorithmns
MIN_VALIDATED_THRESHOLD = 2000

#
# Program parameters
#

# If true, report all on different lines, else show only generated
VERBOSE: bool = False
# If true, fail if source is not found, else skip it
FAIL_ON_NOT_FOUND: bool = True
# If true, regenerate the splits even if they exist,
FORCE_CREATE: bool = False
