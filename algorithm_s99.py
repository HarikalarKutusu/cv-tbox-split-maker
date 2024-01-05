#!/usr/bin/env python3
"""
cv-tbox Diversity Check / Split Maker
Standard Common Voice CorporaCreator algorithm with 99 recordings for a sentences is allowed
"""
###########################################################################
# algorithm-s99.py
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
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
import sys
import shutil
import glob
import multiprocessing as mp
import threading


# External Dependencies
import pandas as pd
import psutil
import corporacreator
from tqdm import tqdm

# Module
from typedef import AlgorithmSpecs, Globals
from lib import LocalCorpus
from lib import df_read, final_report
import conf


#
# Globals
#
HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)
PROC_COUNT: int = psutil.cpu_count(logical=True)  # Full usage
output_lock = threading.Lock()

g = Globals()
aspecs = AlgorithmSpecs(
    src_algo_dir="s1", dst_algo_dir="s99", duplicate_sentence_count=99
)


#
# PROCESS
# Handle one split creation, this is where calculations happen
#


# def corpora_creator_original(lc: str, val_path: str, dst_path: str, duplicate_sentences: int) -> bool:
def corpora_creator_original(val_path: str) -> bool:
    """Processes validated.tsv and create new train, dev, test splits"""
    dst_exppath: str = os.path.join(HERE, "experiments", aspecs.dst_algo_dir)
    # results: list[bool] = []

    src_corpus_dir: str = os.path.split(val_path)[0]
    lc: str = os.path.split(src_corpus_dir)[1]
    ver: str = os.path.split(os.path.split(src_corpus_dir)[0])[1]
    dst_corpus_dir: str = os.path.join(dst_exppath, ver, lc)

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
            [
                "-d",
                temp_path,
                "-f",
                val_path,
                "-s",
                str(aspecs.duplicate_sentence_count),
            ]
        )
        corpus: LocalCorpus = LocalCorpus(args, lc, df_corpus)
        corpus.create()
        corpus.save(temp_path)

        # move required files to destination
        os.makedirs(dst_corpus_dir, exist_ok=True)
        shutil.move(os.path.join(temp_path, lc, "train.tsv"), dst_corpus_dir)
        shutil.move(os.path.join(temp_path, lc, "dev.tsv"), dst_corpus_dir)
        shutil.move(os.path.join(temp_path, lc, "test.tsv"), dst_corpus_dir)
        shutil.rmtree(os.path.join(temp_path, lc))

        res = True

    return res


#
# Main loop for experiments-versions-locales
#


def main() -> None:
    """Original Corpora Creator with -s 99 option for Common Voice Datasets"""

    #
    # Callback
    #

    def pool_callback(res: bool) -> None:
        """Callback to append results and increment bar"""
        pbar.update()
        if res:
            g.processed_cnt += 1
        else:
            g.skipped_nodata += 1

    #
    # Main
    #

    print(
        "=== Original Corpora Creator with -s 99 option for Common Voice Datasets ==="
    )

    # Paths
    src_exppath: str = os.path.join(HERE, "experiments", aspecs.src_algo_dir)
    dst_exppath: str = os.path.join(HERE, "experiments", aspecs.dst_algo_dir)

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

    # schedule mp
    print(
        f"Re-splitting for {g.src_cnt} out of {g.total_cnt} corpora in {PROC_COUNT} processes."
    )
    print(f"Skipping {g.skipped_exists} as they already exist.")

    pbar = tqdm(total=g.src_cnt, unit="Dataset")
    with mp.Pool(PROC_COUNT) as pool:
        for val_path in final_list:
            pool.apply_async(
                corpora_creator_original, args=(val_path,), callback=pool_callback
            )
    pbar.close()

    final_report(g)

    # g.finish_time = datetime.now()
    # g.process_seconds = (g.finish_time - g.start_time).total_seconds()
    # avg_seconds: float = -1
    # avg_seconds_actual: float = -1
    # if g.src_cnt > 0:
    #     avg_seconds = g.process_seconds / g.src_cnt
    # if g.processed_cnt > 0:
    #     avg_seconds_actual = g.process_seconds / g.processed_cnt
    # print("\n" + "-" * 80)
    # print(
    #     f"Finished processing of {g.total_cnt} corpora in {str(g.process_seconds)} secs"
    # )
    # print(
    #     f"Checked: {g.src_cnt}, Skipped (no-data): {g.skipped_nodata}, Actual: {g.processed_cnt}"
    # )
    # print(
    #     f'AVG in CHECKED {dec3(avg_seconds)} secs, '
    #     + f'AVG in ACTUAL {dec3(avg_seconds_actual)} secs, '
    # )


if __name__ == "__main__":
    main()
