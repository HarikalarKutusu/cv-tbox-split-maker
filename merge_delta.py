#!/usr/bin/env python3
"""cv-tbox Diversity Check / Split Maker - Delta Upgrade"""
###########################################################################
# delta_upgrade.py
#
# Get data from current delta and previous version, combine them.
# Put them into experiment directories, and run the s1 algorithm
#
# Use:
# python delta_upgrade.py
#
# This script is part of Common Voice ToolBox Package
#
# github: https://github.com/HarikalarKutusu/cv-tbox-split-maker
# Copyright: (c) Bülent Özden, License: AGPL v3.0
###########################################################################

# Standard Lib
import os
import sys
import glob
import shutil
import multiprocessing as mp
import threading
from datetime import datetime
from typing import Literal, TypeAlias, TypedDict

# External Dependencies
import psutil
import pandas as pd
from tqdm import tqdm

# Module
import conf
from lib import dec3, df_read, df_write

# Globals

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

PROC_LIMIT: int = 4
PROC_COUNT: int = min(PROC_LIMIT, psutil.cpu_count(logical=True))
output_lock = threading.Lock()


class Params(TypedDict):
    """MultiProcessing parameters"""

    lc: str
    old_dir: str
    delta_dir: str
    new_dir: str


def merge_delta_process(params: Params) -> None:
    """Multiprocessing handler for single language delta merge"""
    # create destinations
    lc: str = os.path.split(params["delta_dir"])[-1]
    # create destination full dir
    os.makedirs(params["new_dir"], exist_ok=True)
    # first copy non-delta nature files to full-dir (if they exsist - introduced with v17.0)
    for f in ["validated_sentences", "unvalidated_sentences"]:
        _fpath: str = os.path.join(params["delta_dir"], f"{f}.tsv")
        if os.path.isfile(_fpath):
            shutil.copy2(_fpath, params["new_dir"])
    #
    # Merge data in files with delta nature
    #
    decided_set: set[str] = {}
    for f in ["validated", "invalidated", "clip_durations", "reported"]:
        df_prev: pd.DataFrame = df_read(os.path.join(params["old_dir"], f"{f}.tsv"))
        df_delta: pd.DataFrame = df_read(os.path.join(params["delta_dir"], f"{f}.tsv"))
        # handle any new columns
        if len(set(df_prev.columns) - set(df_delta.columns)) != 0:
            df_prev.reindex(columns=df_delta.columns)
        # merge & sort & save
        df_final: pd.DataFrame = pd.concat([df_prev, df_delta])
        # we don't have duplicates - except in "reported.tsv"
        if f != "reported":
            df_final.drop_duplicates(inplace=True)
        # sort by a field - field depends on file
        _cols: list[str] = list(df_final.columns)
        if "path" in _cols:
            df_final.sort_values(["path"], inplace=True)
        elif "sentence_id" in _cols:
            df_final.sort_values(["sentence_id"], inplace=True)
        elif "clip" in _cols:
            df_final.sort_values(["clip"], inplace=True)
        # write out new data
        df_write(df_final, os.path.join(params["new_dir"], f"{f}.tsv"))
        # keep record of items in validated & invalidated for "other.tsv" calculation
        if f in ["validated", "invalidated"]:
            decided_set = set(list(decided_set) + list(set(df_final["path"].to_list())))
    #
    # Handle "other"
    #
    df_prev: pd.DataFrame = df_read(os.path.join(params["old_dir"], "other.tsv"))
    df_delta: pd.DataFrame = df_read(os.path.join(params["delta_dir"], "other.tsv"))
    # handle any new columns
    if len(set(df_prev.columns) - set(df_delta.columns)) != 0:
        df_prev.reindex(columns=df_delta.columns)
    # merge & dedup & sort => still contains recs which moved to val/inval in new version
    df_final: pd.DataFrame = (
        pd.concat([df_prev, df_delta]).drop_duplicates().sort_values(["path"])
    )
    # only allow those not in new val & inval
    df_final = df_final[~df_final["path"].isin(decided_set)]
    df_write(df_final, os.path.join(params["new_dir"], "other.tsv"))
    # return ğarams
    return params


def main(base_prev_dir: str, base_delta_dir: str) -> None:
    """Main process"""
    start_time: datetime = datetime.now()
    #
    # Checks
    #

    # Check incoming directories
    if not os.path.isdir(base_prev_dir):
        print(
            f"FATAL: Source expanded previous version directory could not be located ({base_prev_dir})"
        )
        return
    if not os.path.isdir(base_delta_dir):
        print(
            f"FATAL: Source expanded delta directory could not be located ({base_delta_dir})"
        )
        return
    delta_release_dirname: str = os.path.split(base_delta_dir)[-1]
    if "delta" not in delta_release_dirname:
        print(f"FATAL: This directory is not for delta release ({base_delta_dir})")
        return

    # Check if delta dir has language data
    delta_lc_dirs: list[str] = glob.glob(os.path.join(base_delta_dir, "*"))
    total_cnt: int = len(delta_lc_dirs)
    if total_cnt == 0:
        print("FATAL: Delta directory does not contain expanded language directories.")
        return

    # Build parameter list - also with some checks
    params_list: list[Params] = []
    existing_list: list[str] = []
    no_prev_list: list[str] = []
    for delta_lc_dir in delta_lc_dirs:
        _lc: str = os.path.split(delta_lc_dir)[-1]
        _old_lc_dir: str = os.path.join(
            conf.CV_EXTRACTED_BASE_DIR, conf.CV_FULL_PREV_VERSION, _lc
        )
        _new_lc_dir: str = delta_lc_dir.replace("-delta-", "-")
        # skip extra processing if prev-lc-dir does not exists
        if not os.path.isdir(_old_lc_dir):
            no_prev_list.append(_lc)
            continue
        # skip if destination exists
        if os.path.isdir(_new_lc_dir) and not conf.FORCE_CREATE:
            existing_list.append(_lc)
            continue
        # now we are OK
        params_list.append(
            Params(
                lc=_lc,
                old_dir=_old_lc_dir,
                delta_dir=delta_lc_dir,
                new_dir=_new_lc_dir,
            )
        )
    if existing_list:
        print(f"SKIPPED [{len(existing_list)}] - EXISTING: {existing_list}")
    if no_prev_list:
        print(f"SKIPPED [{len(no_prev_list)}] - NO OLD VERSION: {no_prev_list}")

    actual_cnt: int = len(params_list)
    if actual_cnt == 0:
        print("No delta datasets to merge!")
        return

    #
    # Get delta .tsv from expanded delta directory (validated, invalidated, other)
    # Get related .tsv from previous version directory
    # concat them and save into base experiments/s1 directory
    #
    print(f"Delta-Merge {actual_cnt} locales out of {total_cnt} PROCS=")

    num_procs: int = max(1, min(PROC_COUNT, actual_cnt))
    chunk_size: int = max(1, min(actual_cnt // 100, actual_cnt // num_procs))
    _cnt: int = 0
    _par: Params

    with mp.Pool(processes=PROC_COUNT) as pool:
        with tqdm(delta_lc_dirs, total=actual_cnt, unit=" Dataset") as pbar:
            for _par in pool.imap_unordered(
                merge_delta_process, params_list, chunksize=chunk_size
            ):
                _cnt += 1
                pbar.write(f"Finished: {_par['lc']}\t[{_cnt}/{actual_cnt}]")
                pbar.update()
    # finalize
    seconds: int = (datetime.now() - start_time).seconds
    print(f"Finished in {seconds} sec - Avg={dec3(seconds/actual_cnt)} sec/locale")


if __name__ == "__main__":
    # [TODO] : use command line args
    # args_src_dir: str = "m:\\DATASETS\\cv\\cv-corpus-16.0-2023-12-06"
    args_prev_dir: str = os.path.join(
        conf.CV_METADATA_BASE_DIR, conf.CV_FULL_PREV_VERSION
    )
    args_delta_dir: str = os.path.join(conf.CV_METADATA_BASE_DIR, conf.CV_DELTA_VERSION)

    main(args_prev_dir, args_delta_dir)
