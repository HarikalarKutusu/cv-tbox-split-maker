#!/usr/bin/env python3
"""
cv-tbox Diversity Check / Split Maker
Standard Common Voice CorporaCreator algorithm which is used to create the default splits
"""
###########################################################################
# algorithm-s1.py
#
# Runs Common Voice Corpora Creator with -s 99 parameter
#
# Ff destination already exists, it is skipped,
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
# [github]
# [copyright]
###########################################################################

# Standard Lib
from argparse import Namespace
import os
import sys
import shutil
import glob
from datetime import datetime, timedelta
from typing import Any
import logging

# External dependencies
import pandas as pd
import av
import corporacreator

# Module
import conf
from lib import Globals, AlgorithmSpecs, LocalCorpus, final_report
from lib import df_read, df_write, dec3

# Get rid of warnings
logging.getLogger("libav").setLevel(logging.ERROR)

# Globals
HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

g = Globals()
aspecs = AlgorithmSpecs(
    src_algo_dir="s1", dst_algo_dir="s1", duplicate_sentence_count=1
)

#
# Constants - TODO These should be arguments
#

# Directories
USE_SOURCE_DATASET_DIR: bool = True
DO_CALC_CLIP_DURATIONS: bool = True

# DF related (for clip durations)
CDUR_COLS: list[str] = ["clip", "duration[ms]"]
CDUR_FN: str = "$clip_durations.tsv"
CDUR_ERR_COLS: list[str] = ["clip", "error"]
CDUR_ERR_FN: str = "$clip_durations_errors.tsv"


#
# Handle one split creation, this is where calculations happen
#


def corpora_creator_original(
    lc: str, val_path: str, dst_path: str, duplicate_sentences: int
) -> bool:
    """Processes validated.tsv and create new train, dev, test splits"""

    # Assume result false
    res: bool = False
    # temp dir
    temp_path: str = os.path.join(HERE, ".temp")

    # call corpora creator with only validated (we don't need others)
    df_corpus: pd.DataFrame = df_read(val_path)

    # Must have records in it
    if df_corpus.shape[0] > 0:
        # create temp dir
        os.makedirs(temp_path, exist_ok=True)

        # handle corpus
        args: Namespace = corporacreator.parse_args(
            ["-d", temp_path, "-f", val_path, "-s", str(duplicate_sentences)]
        )
        corpus: LocalCorpus = LocalCorpus(args, lc, df_corpus)
        corpus.create()
        corpus.save(temp_path)

        # move required files to destination
        os.makedirs(dst_path, exist_ok=True)
        shutil.move(os.path.join(temp_path, lc, "train.tsv"), dst_path)
        shutil.move(os.path.join(temp_path, lc, "dev.tsv"), dst_path)
        shutil.move(os.path.join(temp_path, lc, "test.tsv"), dst_path)
        shutil.rmtree(temp_path)

        res = True

    return res


#
# Main loop for experiments-versions-locales
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


#
# Main loop for experiments-versions-locales
#


def handle_clip_durations():
    """Refresh cklip durations if they do not exist"""
    print("=== REFRESH CLIP DURATIONS ===")
    # remove existing clip durations from older versions
    old_clip_durations: list[str] = glob.glob(
        os.path.join(HERE, "experiments", "**", CDUR_FN), recursive=True
    )
    print(
        f"=== Found {len(old_clip_durations)} files in local files, we will delete older ones..."
    )
    for inx, clip_path in enumerate(old_clip_durations):
        # keep for last version
        if not clip_path.split(os.sep)[-4] == conf.CV_DATASET_VERSION:
            print("Remove:", inx, "/".join(clip_path.split(os.sep)[-4:]))
            os.remove(path=clip_path)
        else:
            print("Skip:", inx, "/".join(clip_path.split(os.sep)[-4:]))
    # recalculate clip durations
    glob_path: str = os.path.join(
        conf.CV_DATASET_BASE_DIR, conf.CV_DATASET_VERSION, "**", "clips"
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


def main() -> None:
    """
    Original Corpora Creator with -s 1 option for Common Voice Datasets (if splits are not provided)
    """
    print(
        "=== Original Corpora Creator with -s 1 option for Common Voice Datasets (if splits are not provided) ==="
    )

    # Copy source experiment tree to destination experiment
    src_exppath: str = os.path.join(HERE, "experiments", aspecs.src_algo_dir)
    dst_exppath: str = os.path.join(HERE, "experiments", aspecs.dst_algo_dir)

    # Calculate clip durations?
    if DO_CALC_CLIP_DURATIONS:
        handle_clip_durations()

    # Do we want to copy the .tsv files from original expanded datasets?
    if USE_SOURCE_DATASET_DIR:
        # copy all .tsv files while forming structure
        print("=== COPY .TSV FILES FROM DATASETS ===")
        copyto_corpus_dir: str = os.path.join(src_exppath, conf.CV_DATASET_VERSION)
        os.makedirs(name=copyto_corpus_dir, exist_ok=True)
        shutil.copytree(
            src=os.path.join(conf.CV_DATASET_BASE_DIR, conf.CV_DATASET_VERSION),
            dst=copyto_corpus_dir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("*.mp3"),
        )

    # Get total for progress display
    all_validated: "list[str]" = glob.glob(
        os.path.join(src_exppath, "**", "validated.tsv"), recursive=True
    )
    print(
        f"Re-splitting for {len(all_validated)} corpora... Wait for final structure is formed..."
    )
    print()  # extra line is for progress line

    # For each corpus
    g.start_time = datetime.now()
    g.total_cnt = len(all_validated)
    g.processed_cnt = 0  # count of corpora checked

    for val_path in all_validated:
        src_corpus_dir: str = os.path.split(val_path)[0]
        lc: str = os.path.split(src_corpus_dir)[1]
        ver: str = os.path.split(os.path.split(src_corpus_dir)[0])[1]
        dst_corpus_dir: str = os.path.join(dst_exppath, ver, lc)

        g.processed_cnt += 1
        if conf.VERBOSE:
            print(f"\n=== Processing {g.processed_cnt}/{g.total_cnt} => {ver} - {lc}")
        else:
            print("\033[F" + " " * 80)
            print(f"\033[FProcessing {g.processed_cnt}/{g.total_cnt} => {ver} - {lc}")

        if not conf.FORCE_CREATE and os.path.isfile(
            os.path.join(dst_corpus_dir, "train.tsv")
        ):
            # Already there and is not forced to recreate, so skip
            g.skipped_exists += 1
        else:
            if not corpora_creator_original(  # df might be empty, thus returns false
                lc=lc,
                val_path=val_path,
                dst_path=dst_corpus_dir,
                duplicate_sentences=aspecs.duplicate_sentence_count,
            ):
                g.skipped_exists += 1
            print()

    final_report(g)

    # g.finish_time = datetime.now()
    # g.process_seconds = (g.finish_time - g.start_time).total_seconds()
    # avg_seconds: float = g.process_seconds / g.total_cnt
    # cnt_really_processed: int = g.processed_cnt - g.skipped_exists - g.skipped_nodata
    # avg_seconds_new: float = -1
    # if cnt_really_processed > 0:
    #     avg_seconds_new = g.process_seconds / cnt_really_processed
    # print("\n" + "-" * 80)
    # print(
    #     f"Finished processing of {g.total_cnt} corpora in {str(g.process_seconds)} secs,"
    #     + f"avg duration {dec3(avg_seconds)} secs"
    # )
    # print(f"Processed: {g.processed_cnt}, Skipped: {g.skipped_exists}, New: {cnt_really_processed}")
    # if cnt_really_processed > 0:
    #     print(
    #         f'Avg. time new split creation: {dec3(avg_seconds_new)} secs'
    #     )


if __name__ == "__main__":
    main()
