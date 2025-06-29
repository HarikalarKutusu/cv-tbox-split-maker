#!/usr/bin/env python3
"""
cv-tbox Diversity Check / Split Maker
Standard Common Voice CorporaCreator algorithm which is used to create the default splits
"""
###########################################################################
# algorithm-s1.py
#
# Runs Common Voice Corpora Creator with -s 1 parameter
# This is required if the dataset does not include train/dev/test.tsv files
#
# If destination already exists, it is skipped,
# else Corpora Creator is run.
#
# The result will only include train, dev, tes tsv files.
#
# Uses multiprocessing with N-1 cores.
#
# Use:
# python algorithm-s99.py
#
# This script is part of Common Voice ToolBox Package
#
# github: https://github.com/HarikalarKutusu/cv-tbox-split-maker
# Copyright: (c) Bülent Özden, License: AGPL v3.0
###########################################################################

# Standard Lib
from argparse import Namespace
from datetime import datetime, timedelta
from typing import Any
import os
import sys
import shutil
import glob
import logging
import multiprocessing as mp
import threading

# External dependencies
from tqdm import tqdm
from corporacreator import parse_args
import pandas as pd
import psutil
import av

# Module
import conf
from typedef import AlgorithmSpecs, Globals
from lib import (
    LocalCorpus,
    df_read,
    df_write,
    final_report,
    mp_optimize_params,
    remove_deleted_users,
    sort_by_largest_file,
)

# Get rid of warnings
logging.getLogger("libav").setLevel(logging.ERROR)

# Globals
HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)
PROC_COUNT: int = psutil.cpu_count(logical=True) or 1  # Full usage
output_lock = threading.Lock()

g = Globals()
algo_specs = AlgorithmSpecs(
    src_algo_dir="s1",
    dst_algo_dir="s1",
    duplicate_sentence_count=1,
)

#
# Constants - TODO These should be arguments
#

# DF related (for clip durations)
CDUR_COLS: list[str] = ["clip", "duration[ms]"]
CDUR_FN: str = "$clip_durations.tsv"
CDUR_ERR_COLS: list[str] = ["clip", "error"]
CDUR_ERR_FN: str = "$clip_durations_errors.tsv"


#
# Handle one split creation, this is where calculations happen
#


# def corpora_creator_original(
#     lc: str, val_path: str, dst_path: str, duplicate_sentences: int
# ) -> bool:
#     """Processes validated.tsv and create new train, dev, test splits"""

#     # Assume result false
#     res: bool = False
#     # temp dir
#     temp_path: str = os.path.join(HERE, ".temp")

#     # call corpora creator with only validated (we don't need others)
#     df_corpus: pd.DataFrame = df_read(val_path)

#     # Must have records in it
#     if df_corpus.shape[0] > 0:
#         # create temp dir
#         os.makedirs(temp_path, exist_ok=True)

#         # handle corpus
#         args: Namespace = parse_args(
#             ["-d", temp_path, "-f", val_path, "-s", str(duplicate_sentences)]
#         )
#         corpus: LocalCorpus = LocalCorpus(args, lc, df_corpus)
#         corpus.create()
#         corpus.save(temp_path)

#         # move required files to destination
#         os.makedirs(dst_path, exist_ok=True)
#         shutil.move(os.path.join(temp_path, lc, "train.tsv"), dst_path)
#         shutil.move(os.path.join(temp_path, lc, "dev.tsv"), dst_path)
#         shutil.move(os.path.join(temp_path, lc, "test.tsv"), dst_path)
#         shutil.rmtree(temp_path)

#         res = True

#     return res

#
# PROCESS
# Handle one split creation, this is where calculations happen
#


# def corpora_creator_original(lc: str, val_path: str, dst_path: str, duplicate_sentences: int) -> bool:
def corpora_creator_original(val_path: str) -> bool:
    """Processes validated.tsv and create new train, dev, test splits"""
    dst_exppath: str = os.path.join(
        conf.SM_DATA_DIR, "experiments", algo_specs.dst_algo_dir
    )
    # results: list[bool] = []

    src_corpus_dir: str = os.path.split(val_path)[0]
    lc: str = os.path.split(src_corpus_dir)[1]
    ver: str = os.path.split(os.path.split(src_corpus_dir)[0])[1]
    dst_corpus_dir: str = os.path.join(dst_exppath, ver, lc)

    # temp dir
    temp_path: str = os.path.join(HERE, ".temp", ver, lc)

    # call corpora creator with only validated (we don't need others)
    df_corpus: pd.DataFrame = df_read(val_path)
    num_original: int = df_corpus.shape[0]

    # Must have records in it, else no go
    if num_original == 0:
        return False

    # Remove users who requested data deletion
    df_corpus = remove_deleted_users(df_corpus)
    if num_original != df_corpus.shape[0] and conf.VERBOSE:
        print(
            f"\nUSER RECORDS DELETED FROM VALIDATED {ver}-{lc} = {num_original - df_corpus.shape[0]}"
        )

    # Here, it has records in it
    # create temp dir
    os.makedirs(temp_path, exist_ok=True)

    # handle corpus
    cc_args: Namespace = parse_args(
        [
            "-d",
            temp_path,
            "-f",
            val_path,
            "-s",
            str(algo_specs.duplicate_sentence_count),
        ]
    )
    corpus: LocalCorpus = LocalCorpus(cc_args, lc, df_corpus)
    corpus.create()
    corpus.save(temp_path)

    # move required files to destination
    os.makedirs(dst_corpus_dir, exist_ok=True)

    # copy to original directory (source) in case of merge
    shutil.copy(os.path.join(temp_path, lc, "train.tsv"), src_corpus_dir)
    shutil.copy(os.path.join(temp_path, lc, "dev.tsv"), src_corpus_dir)
    shutil.copy(os.path.join(temp_path, lc, "test.tsv"), src_corpus_dir)

    # move to algorithm directory (destination)
    shutil.move(os.path.join(temp_path, lc, "train.tsv"), dst_corpus_dir)
    shutil.move(os.path.join(temp_path, lc, "dev.tsv"), dst_corpus_dir)
    shutil.move(os.path.join(temp_path, lc, "test.tsv"), dst_corpus_dir)

    shutil.rmtree(os.path.join(temp_path, lc))

    return True


#
# Clip Durations
#


# Main Loop for Clips
def build_clip_durations_table(srcdir):
    """
    Creates clip durations table from audio files in a directory.
    Only called when it is not allready supplied.
    """
    start: datetime = datetime.now()
    # get list
    mp3list: list[str] = glob.glob(os.path.join(srcdir, "*.mp3"))
    mp3list.sort()
    num_files: int = len(mp3list)
    log_step: int = int(num_files / 10)
    if log_step == 0:
        log_step = 1
    # process list
    cnt: int = 0
    skipped: int = 0
    total_dur: float = 0
    perc = 0
    # Start display
    print(f'Creating {CDUR_FN} table for {num_files} clips into "{srcdir}"')
    print(
        "+" * perc
        + "." * (100 - perc)
        + f" {perc}% - {cnt}/{num_files} => {0.00} hours."
    )
    data: list[Any] = []
    data_err: list[Any] = []
    a: Any = None
    for fn in mp3list:
        cnt += 1
        perc: int = int(100 * cnt / num_files + 0.5)
        if os.path.getsize(fn) == 0:
            print(f"ERROR: Zero filesize  - {fn}")
            skipped += 1
            data_err.append([os.path.split(fn)[-1], "zero_filesize"])
            continue  # skip if filesize is 0

        err: bool = False
        try:
            a = av.open(fn)
        except ValueError as e:
            print(f"ERROR: During opening - {fn}")
            data_err.append([os.path.split(fn)[-1], "could_not_open"])
            skipped += 1
            err: bool = True

        if not err and a:
            file_duration: float = (a.duration) / 1000000
            total_dur += file_duration
            if cnt % log_step == 0:
                print(
                    "+" * perc
                    + "." * (100 - perc)
                    + f" {perc}% - {cnt}/{num_files} => {round(total_dur/3600,2)} hours."
                )
            # add to list
            data.append([os.path.split(fn)[-1], file_duration])

    # finish
    finish: datetime = datetime.now()
    pduration: timedelta = finish - start
    pduration_sec: int = int(pduration.total_seconds())
    if skipped > 0:
        print(f"Skipped {skipped} files due to errors.")
    print(
        f"Finished {num_files} files in {pduration} sec, avg= {pduration_sec/num_files}."
    )
    print(
        f"Total audio duration {round(total_dur/3600,2)} hours, avg. duration= {total_dur/(num_files-skipped)} sec."
    )
    # Build dataframe and save
    df: pd.DataFrame = pd.DataFrame(data, columns=CDUR_COLS).reset_index(drop=True)
    df_write(df, fpath=os.path.join(srcdir, CDUR_FN))
    if len(data_err) > 0:
        df_err: pd.DataFrame = pd.DataFrame(
            data_err, columns=CDUR_ERR_COLS
        ).reset_index(drop=True)
        df_write(df, fpath=os.path.join(srcdir, CDUR_ERR_FN))


def handle_clip_durations():
    """Refresh cklip durations if they do not exist"""
    print("=== REFRESH CLIP DURATIONS ===")
    # remove existing clip durations from older versions
    old_clip_durations: list[str] = glob.glob(
        os.path.join(conf.SM_DATA_DIR, "experiments", "**", CDUR_FN), recursive=True
    )
    print(
        f"=== Found {len(old_clip_durations)} files in local files, we will delete older ones..."
    )
    for inx, clip_path in enumerate(old_clip_durations):
        # keep for last version
        if not clip_path.split(os.sep)[-4] == conf.CV_FULL_VERSION:
            print("Remove:", inx, "/".join(clip_path.split(os.sep)[-4:]))
            os.remove(path=clip_path)
        else:
            print("Skip:", inx, "/".join(clip_path.split(os.sep)[-4:]))
    # recalculate clip durations
    glob_path: str = os.path.join(
        conf.CV_EXTRACTED_BASE_DIR, conf.CV_FULL_VERSION, "**", "clips"
    )
    print(f"Searching clips dirs with {glob_path}")
    clips_dirs: list[str] = glob.glob(glob_path, recursive=False)
    print(f"=== Processing {len(clips_dirs)} locales (files created in data source)")
    for inx, clips_dir in enumerate(clips_dirs):
        # create only if file does not exists
        if not os.path.isfile(os.path.join(clips_dir, CDUR_FN)):
            build_clip_durations_table(clips_dir)
        else:
            print("Skip:", inx, "/".join(clips_dir.split(os.sep)[-4:]))


#
# Main loop for experiments-versions-locales
#


def main(collect: bool, calc_durations: bool) -> None:
    """
    Original Corpora Creator with -s 1 option for Common Voice Datasets (if splits are not provided)
    """
    print(
        "=== Original Corpora Creator with -s 1 option for Common Voice Datasets (if splits are not provided, merge worklow) ==="
    )

    #
    # Main
    #

    # Copy source experiment tree to destination experiment
    src_exppath: str = os.path.join(
        conf.SM_DATA_DIR, "experiments", algo_specs.src_algo_dir
    )
    dst_exppath: str = os.path.join(
        conf.SM_DATA_DIR, "experiments", algo_specs.dst_algo_dir
    )

    # Calculate clip durations?
    if calc_durations:
        handle_clip_durations()

    # Do we want to copy the .tsv files from original expanded datasets?
    if collect:
        # copy all .tsv files while forming structure
        print("=== COPY .TSV FILES FROM DATASETS ===")
        copyto_corpus_dir: str = os.path.join(src_exppath, conf.CV_FULL_VERSION)
        os.makedirs(name=copyto_corpus_dir, exist_ok=True)
        shutil.copytree(
            src=os.path.join(conf.CV_EXTRACTED_BASE_DIR, conf.CV_FULL_VERSION),
            dst=copyto_corpus_dir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("*.mp3"),
        )

    # # Get total for progress display
    # all_validated: "list[str]" = glob.glob(
    #     os.path.join(src_exppath, "**", "validated.tsv"), recursive=True
    # )
    # print(
    #     f"Re-splitting for {len(all_validated)} corpora... Wait for final structure is formed..."
    # )
    # print()  # extra line is for progress line

    # # For each corpus
    # g.start_time = datetime.now()
    # g.total_cnt = len(all_validated)
    # g.processed_cnt = 0  # count of corpora checked

    # for val_path in all_validated:
    #     src_corpus_dir: str = os.path.split(val_path)[0]
    #     lc: str = os.path.split(src_corpus_dir)[1]
    #     ver: str = os.path.split(os.path.split(src_corpus_dir)[0])[1]
    #     dst_corpus_dir: str = os.path.join(dst_exppath, ver, lc)

    #     g.processed_cnt += 1
    #     if conf.VERBOSE:
    #         print(f"\n=== Processing {g.processed_cnt}/{g.total_cnt} => {ver} - {lc}")
    #     else:
    #         print("\033[F" + " " * 80)
    #         print(f"\033[FProcessing {g.processed_cnt}/{g.total_cnt} => {ver} - {lc}")

    #     if not conf.FORCE_CREATE and os.path.isfile(
    #         os.path.join(dst_corpus_dir, "train.tsv")
    #     ):
    #         # Already there and is not forced to recreate, so skip
    #         g.skipped_exists += 1
    #     else:
    #         if not corpora_creator_original(  # df might be empty, thus returns false
    #             lc=lc,
    #             val_path=val_path,
    #             dst_path=dst_corpus_dir,
    #             duplicate_sentences=aspecs.duplicate_sentence_count,
    #         ):
    #             g.skipped_exists += 1
    #         print()

    # final_report(g)

    # Get total for progress display
    all_validated: "list[str]" = glob.glob(
        os.path.join(src_exppath, "**", "validated.tsv"), recursive=True
    )

    # For each corpus
    g.start_time = datetime.now()
    final_list: list[str] = []

    # clean unneeded/skipped
    if conf.FORCE_CREATE:
        final_list = all_validated
    else:
        for p in all_validated:
            src_corpus_dir: str = os.path.split(p)[0]
            lc: str = os.path.split(src_corpus_dir)[1]
            ver: str = os.path.split(os.path.split(src_corpus_dir)[0])[1]
            dst_corpus_dir: str = os.path.join(dst_exppath, ver, lc)
            if os.path.isfile(os.path.join(dst_corpus_dir, "train.tsv")):
                g.skipped_exists += 1
            else:
                final_list.append(p)

    g.total_cnt = len(all_validated)
    g.src_cnt = len(final_list)

    # MP optimization
    final_list = sort_by_largest_file(final_list)
    final_list = mp_optimize_params(final_list, PROC_COUNT)

    # schedule mp
    chunk_size: int = min(
        10, g.src_cnt // PROC_COUNT + 0 if g.src_cnt % PROC_COUNT == 0 else 1
    )
    print(f"Skipped {g.skipped_exists} as they already exist.")
    print(
        f"Splitting {g.src_cnt} out of {g.total_cnt} corpora PROCS={PROC_COUNT} chunk-size={chunk_size}."
    )

    with mp.Pool(PROC_COUNT) as pool:
        with tqdm(total=g.src_cnt) as pbar:
            for res in pool.imap_unordered(
                corpora_creator_original, final_list, chunksize=chunk_size
            ):
                pbar.update()
                if res:
                    g.processed_cnt += 1
                else:
                    g.skipped_nodata += 1

    # remove temp directory structure
    # _ = [shutil.rmtree(d) for d in glob.glob(os.path.join(HERE, ".temp", "*"), recursive=False)]

    final_report(g)


if __name__ == "__main__":
    args: list[str] = sys.argv
    arg_collect: bool = "--collect" in args
    arg_calc_durations: bool = "--durations" in args

    main(arg_collect, arg_calc_durations)
