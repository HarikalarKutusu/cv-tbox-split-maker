#!/usr/bin/env python3
"""cv-tbox Diversity Check / TSV File Extractor"""
###########################################################################
# extract.py
#
# From a directory containing downloaded Common Voice dataset files
# extracts only the .tsv files or all files into another location.
#
# Use:
# python extract.py
#
# This script is part of Common Voice ToolBox Package
#
# github: https://github.com/HarikalarKutusu/common-voice-diversity-check
# Copyright: (c) Bülent Özden, License: AGPL v3.0
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

MINIMAL_PROCS: int = 2


def extract_all_process(p: str) -> int:
    """Multiprocessing handler for full extraction"""
    with tarfile.open(p) as tar:
        tar.extractall(conf.CV_DATASET_BASE_DIR)
    return 1


def extract_tsv_process(p: str) -> int:
    """Multiprocessing handler for extraction"""
    pat: re.Pattern[str] = re.compile(r"^.*\.tsv")
    with tarfile.open(p) as tar:
        tar.extractall(
            conf.CV_DATASET_BASE_DIR,
            members=[m for m in tar.getmembers() if pat.search(m.name)],
        )
    return 1


def main(extract_all: bool = False, forced: bool = False) -> None:
    """Main process"""

    # get compressed files list
    all_files: list[str] = glob.glob(
        os.path.join(conf.CV_COMPRESSED_BASE_DIR, conf.CV_DATASET_VERSION, "*")
    )
    total_cnt: int = len(all_files)

    start_time: datetime = datetime.now()
    src_files: list[str] = all_files if conf.FORCE_CREATE or forced else []

    dst_check: str = os.path.join(conf.CV_DATASET_BASE_DIR, conf.CV_DATASET_VERSION)
    os.makedirs(dst_check, exist_ok=True)

    if extract_all:
        # remove already extracted ones from the list
        if not (conf.FORCE_CREATE or forced):
            for p in all_files:
                lc: str = (
                    os.path.split(p)[-1]
                    .replace(conf.CV_DATASET_VERSION + "-", "")
                    .replace(".tar.gz", "")
                    .replace(".tar", "")
                )
                dst_check = os.path.join(dst_check, lc, "clips")
                if conf.FORCE_CREATE or not os.path.isdir(dst_check):
                    src_files.append(p)

        src_cnt: int = len(src_files)

        # Low number of cores only to prevent HDD trashing
        proc_count: int = min(MINIMAL_PROCS, psutil.cpu_count(logical=False))

        print(f"Extracting ALL files from {src_cnt}/{total_cnt} compressed datasets")
        if conf.FORCE_CREATE:
            print("Expanding even the destination exists (force_create)")
        elif total_cnt > src_cnt:
            print(f"Skipping {total_cnt - src_cnt} already extracted datasets")

        chunk_size: int = src_cnt // proc_count + 0 if src_cnt % proc_count == 0 else 1

        with mp.Pool(proc_count) as pool:
            with tqdm(total=src_cnt) as pbar:
                for _ in pool.imap_unordered(
                    extract_tsv_process, src_files, chunksize=chunk_size
                ):
                    pbar.update()

    else:
        # remove already extracted ones from the list
        if not (conf.FORCE_CREATE or forced):
            for p in all_files:
                lc: str = (
                    os.path.split(p)[-1]
                    .replace(conf.CV_DATASET_VERSION + "-", "")
                    .replace(".tar.gz", "")
                    .replace(".tar", "")
                )
                dst_check = os.path.join(dst_check, lc, "validated.tsv")
                if conf.FORCE_CREATE or not os.path.isfile(dst_check):
                    src_files.append(p)

        src_cnt: int = len(src_files)

        # Real cores only to prevent excessive HDD head movements
        proc_count: int = psutil.cpu_count(logical=False)

        print(
            f"Extracting only .TSV files from {src_cnt}/{total_cnt} compressed datasets"
        )
        if conf.FORCE_CREATE:
            print("Expanding even the destination exists (force_create)")
        elif total_cnt > src_cnt:
            print(f"Skipping {total_cnt - src_cnt} already extracted datasets")

        chunk_size: int = src_cnt // proc_count + 0 if src_cnt % proc_count == 0 else 1

        with mp.Pool(proc_count) as pool:
            with tqdm(total=src_cnt) as pbar:
                for _ in pool.imap_unordered(
                    extract_tsv_process, src_files, chunksize=chunk_size
                ):
                    pbar.update()

    # finalize
    dur: int = (datetime.now() - start_time).seconds
    print(f"Finished in {dur} sec - Avg={dec3(dur/src_cnt)} sec/dataset")


if __name__ == "__main__":
    args: list[str] = sys.argv
    arg_all: bool = "--all" in args
    arg_force: bool = "--force" in args
    main(arg_all, arg_force)
