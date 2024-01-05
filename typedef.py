""" cv-tbox Split Creator - Type Definitions """

# Standard Lib
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Globals:  # pylint: disable=too-many-instance-attributes
    """Class to keep globals in one place"""

    total_cnt: int = 0  # total count of corpora
    src_cnt: int = 0  # subset of total_count to be processed (used by some algorithmns)

    start_time: datetime = datetime.now()
    skipped_exists: int = 0  # skipped befcause the destination already exists
    skipped_nosupport: int = 0  # skipped because not supported (Whisper/Voxpopuli)
    skipped_small: int = 0  # skipped because very few data to work on
    skipped_nodata: int = 0  # skipped because no data / not enough data to process
    processed_cnt: int = 0  # counter for corpora processed

    large_dataset_cnt: int = 0
    medium_dataset_cnt: int = 0
    tiny_dataset_cnt: int = 0


@dataclass
class AlgorithmSpecs:  # pylint: disable=too-many-instance-attributes
    """Class to keep algorithm specs in one place"""

    src_algo_dir: str = "s1"
    dst_algo_dir: str = ""

    duplicate_sentence_count: int = 1

    train_percentage: float = 80.0
    dev_percentage: float = 10.0
    test_percentage: float = 10.0

    max_test_user: float = 50.0
    max_dev_user: float = 25.0


@dataclass
class AlgorithmResults:  # pylint: disable=too-many-instance-attributes
    """Class to keep algorithm specs in one place"""

    ver: str = ""
    lc: str = ""

    processed: int = 0
    skipped_nodata: int = 0
    skipped_small: int = 0
    large: int = 0
    medium: int = 0
    tiny: int = 0
