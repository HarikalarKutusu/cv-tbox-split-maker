""" cv-tbox Split Maker - Configuration File """

import os
import sys
import platform

is_windows: bool = os.name == "nt"
is_linux: bool = sys.platform == "Linux"
is_wsl: bool = "WSL" in platform.uname().release

# Drive prefixes - to support running on different systems

# DRV_COMPRESSED: downloaded compressed datasets (e.g. slow backup drive)
# DRV_MEDIA: used to expand clips in hierarchical structure (e.g. large HDD drive)
# DRV_WORK: Your actual work area for hierarchical clips (e.g. speedy SSD drive)
# DRV_METADATA: Your actual work area for metadata (e.g. speedy SSD drive)
if is_windows:
    DRV_COMPRESSED: str = "V:\\"    # downloaded compressed datasets (e.g. slow backup drive)
    DRV_MEDIA: str = "M:\\"         # used to expand clips in hierarchical structure
    DRV_WORK: str = "T:\\"
    DRV_METADATA: str = "T:\\"
elif is_wsl:
    DRV_COMPRESSED: str = "/mnt/v"
    DRV_MEDIA: str = "/mnt/m"
    DRV_WORK: str = "/mnt/t"
    DRV_METADATA: str = "/mnt/t"
elif is_linux:
    DRV_COMPRESSED: str = "/mnt/v"
    DRV_MEDIA: str = "/mnt/m"
    DRV_WORK: str = "/mnt/t"
    DRV_METADATA: str = "/mnt/t"
else:
    print("FATAL: Unsupported system!")
    sys.exit(-1)

# Locations

CV_COMPRESSED_BASE_DIR: str = os.path.join(DRV_COMPRESSED, "DATASETS", "VOICE", "cv")

CV_DS_MEDIA_DIR: str = os.path.join(DRV_MEDIA, "DATASETS", "cv")
CV_DS_WORK_DIR: str = os.path.join(DRV_WORK, "DATASETS", "cv")
CV_DS_ALL_CLIPS_DIR: str = os.path.join(CV_DS_MEDIA_DIR, "clips")

CV_DS_METADATA_DIR: str = os.path.join(DRV_METADATA, "METADATA", "cv")


# Latest Version to work on

CV_DATASET_VERSION: str = "cv-corpus-17.0-2024-03-15"

# Language Codes we fully work on
# These datasets will fully be expanded regardless of --all setting
# Into the DRV_WORK drive - in addition to DRV_MEDIA

CV_FULL_LC_LIST: list[str] = ["tr"]

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
