#!/usr/bin/env python3
"""cv-tbox Diversity Check / TSV File Extractor"""
###########################################################################
# extract_tsv.py
#
# From a directory containing downloaded Common Voice dataset files
# extracts only the .tsv files into another location.
#
# Use:
# python extract_tsv.py
#
# This script is part of Common Voice ToolBox Package
#
# [github]
# [copyright]
###########################################################################

# Standard Lib
import os
import sys
import re
import glob
import tarfile
from datetime import datetime
import multiprocessing as mp

# External Dependencies
import psutil
from tqdm import tqdm

# Module
import conf
from lib import dec3

# Globals

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

PROC_COUNT: int = psutil.cpu_count(logical=True)  # Full usage


def extract_process(p: str) -> int:
    """Multiprocessing handler for extraction"""
    pat: re.Pattern[str] = re.compile(r"^.*\.tsv")
    with tarfile.open(p) as tar:
        tar.extractall(
            conf.CV_DATASET_BASE_DIR,
            members=[m for m in tar.getmembers() if pat.search(m.name)],
        )
    return 1


def main() -> None:
    """Main process"""

    # get compressed files list
    all_files: list[str] = glob.glob(
        os.path.join(conf.CV_COMPRESSED_BASE_DIR, conf.CV_DATASET_VERSION, "*")
    )
    total_cnt: int = len(all_files)

    # remove already extracted ones from the list
    src_files: list[str] = []
    for p in all_files:
        lc: str = (
            os.path.split(p)[-1]
            .replace(conf.CV_DATASET_VERSION + "-", "")
            .replace(".tar.gz", "")
            .replace(".tar", "")
        )
        dst_val: str = os.path.join(
            conf.CV_DATASET_BASE_DIR, conf.CV_DATASET_VERSION, lc, "validated.tsv"
        )
        if conf.FORCE_CREATE or not os.path.isfile(dst_val):
            src_files.append(p)
    src_cnt: int = len(src_files)

    print(f"Extracting .TSV files from {src_cnt}/{total_cnt} compressed datasets")
    if conf.FORCE_CREATE:
        print("Expanding even the destination exists (force_create)")
    elif total_cnt > src_cnt:
        print(f"Skipping {total_cnt - src_cnt} already extracted datasets")

    start_time: datetime = datetime.now()
    chunk_size: int = src_cnt // PROC_COUNT + 0 if src_cnt % PROC_COUNT == 0 else 1

    with mp.Pool(PROC_COUNT) as pool:
        with tqdm(total=src_cnt) as pbar:
            for _ in pool.imap_unordered(
                extract_process, src_files, chunksize=chunk_size
            ):
                pbar.update()

    # finalize
    dur: int = (datetime.now() - start_time).seconds
    print(f"Finished in {dur} sec - Avg={dec3(dur/src_cnt)} sec/dataset")


if __name__ == "__main__":
    main()
