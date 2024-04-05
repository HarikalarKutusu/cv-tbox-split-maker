#!/usr/bin/env python3
"""
cv-tbox Diversity Check / Split Maker
Standard Common Voice CorporaCreator algorithm with 99 recordings for a sentences is allowed
"""
###########################################################################
# algorithm-s99.py
#
# Runs Common Voice Corpora Creator with -s 5 parameter
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
# github: https://github.com/HarikalarKutusu/common-voice-diversity-check
# Copyright: (c) Bülent Özden, License: AGPL v3.0
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
from tqdm import tqdm
from corporacreator import parse_args
import pandas as pd
import psutil

# Module
from typedef import AlgorithmSpecs, Globals
from lib import LocalCorpus
from lib import df_read, final_report, remove_deleted_users
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
    src_algo_dir="s1", dst_algo_dir="s5", duplicate_sentence_count=5
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
        print(f"\nUSER RECORDS DELETED FROM VALIDATED {ver}-{lc} = {num_original - df_corpus.shape[0]}")

    # Here, it has records in it
    # create temp dir
    os.makedirs(temp_path, exist_ok=True)

    # handle corpus
    args: Namespace = parse_args(
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

    return True


#
# Main loop for experiments-versions-locales
#


def main() -> None:
    """Original Corpora Creator with -s 5 option for Common Voice Datasets"""

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

    chunk_size: int = min(10, g.src_cnt // PROC_COUNT + 0 if g.src_cnt % PROC_COUNT == 0 else 1)

    with mp.Pool(PROC_COUNT) as pool:
        with tqdm(total=g.src_cnt) as pbar:
            for result in pool.imap_unordered(
                corpora_creator_original, final_list, chunksize=chunk_size
            ):
                pool_callback(result)

    # remove temp directory structure
    # _ = [shutil.rmtree(d) for d in glob.glob(os.path.join(HERE, ".temp", "*"), recursive=False)]

    final_report(g)


if __name__ == "__main__":
    main()
