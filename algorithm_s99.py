#!/usr/bin/env python3
""" Standard Common Voice CorporaCreator algorithm with 99 recordings for a sentences is allowed """

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
from dataclasses import dataclass
import os
import sys
import shutil
import glob
import csv
from datetime import datetime, timedelta

import multiprocessing as mp
import threading

# External Dependencies
import numpy as np
import pandas as pd
import progressbar
import corporacreator
import psutil


HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

#
# Globals
#


@dataclass
class Globals:
    """Class to keep globals in a place"""

    total_cnt: int = 0
    src_cnt: int = 0
    skipped_exists: int = 0
    skipped_nodata: int = 0
    processed_cnt: int = 0
    pbar: progressbar.ProgressBar = progressbar.ProgressBar()


g = Globals()
PROC_COUNT: int = psutil.cpu_count(logical=True)  # Full usage
output_lock = threading.Lock()


#
# Constants - TODO These should be arguments
#

DUPLICATE_SENTENCE_COUNT: int = 99

# Directories
SRC_ALGO_DIR: str = "s1"
DST_ALGO_DIR: str = "s99"

# Program parameters
VERBOSE: bool = (
    False  # If true, report all on different lines, else show only generated
)
FAIL_ON_NOT_FOUND: bool = True  # If true, fail if source is not found, else skip it
FORCE_CREATE: bool = False  # If true, regenerate the splits even if they exist

#
# DataFrame file read-write
#


def df_read(fpath: str) -> pd.DataFrame:
    """Read a tsv file into a dataframe"""
    if not os.path.isfile(fpath):
        print(f"FATAL: File {fpath} cannot be located!")
        if FAIL_ON_NOT_FOUND:
            sys.exit(1)

    df: pd.DataFrame = pd.read_csv(
        fpath,
        sep="\t",
        parse_dates=False,
        engine="python",
        encoding="utf-8",
        on_bad_lines="skip",
        quotechar='"',
        quoting=csv.QUOTE_NONE,
    )
    return df


def df_write(df: pd.DataFrame, fpath: str) -> None:
    """Write dataframe to a tsv file"""
    df.to_csv(
        fpath,
        header=True,
        index=False,
        encoding="utf-8",
        sep="\t",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )


#
# Adapted from original Corpora Creator - removed unneeded features
# - Removed logger
# - No need to re-partition (we already have validated)
# - No need to preprocess (s1 already preprocessed the data)
# - Create only train, dev, test
#


#
# Sample Size Calculation, taken from CorporaCreaotr repo (statistics.py)
#
def calc_sample_size(population_size: int) -> float:
    """Calculates the sample size.

    Calculates the sample size required to draw from a population size `population_size`
    with a confidence level of 99% and a margin of error of 1%.

    Args:
    population_size (int): The population size to draw from.
    """
    margin_of_error: float = 0.01
    fraction_picking: float = 0.50
    z_score: float = 2.58  # Corresponds to confidence level 99%
    numerator: float = (z_score**2 * fraction_picking * (1 - fraction_picking)) / (
        margin_of_error**2
    )
    denominator: float = 1 + (
        z_score**2 * fraction_picking * (1 - fraction_picking)
    ) / (margin_of_error**2 * population_size)
    return numerator / denominator


class LocalCorpus:
    """Corpus representing a Common Voice datasets for a given locale.
    Args:
      args ([str]): Command line parameters as list of strings
      locale (str): Locale this :class:`corporacreator.Corpus` represents
      corpus_data (:class:`pandas.DataFrame`): `pandas.DataFrame` Containing the corpus data
    Attributes:
        args ([str]): Command line parameters as list of strings
        locale (str): Locale of this :class:`corporacreator.Corpus`
        corpus_data (:class:`pandas.DataFrame`): `pandas.DataFrame` Containing the corpus data
    """

    def __init__(self, args, locale, corpus_data):
        self.args = args
        self.locale = locale
        self.corpus_data = corpus_data

    def create(self):
        """Creates a :class:`corporacreator.Corpus` for `self.locale`."""
        self._post_process_valid_data()

    def _post_process_valid_data(self):
        # Remove duplicate sentences while maintaining maximal user diversity at the frame's start (TODO: Make addition of user_sentence_count cleaner)
        speaker_counts = self.validated["client_id"].value_counts()
        speaker_counts = speaker_counts.to_frame().reset_index()
        speaker_counts.columns = ["client_id", "user_sentence_count"]
        self.validated = self.validated.join(
            speaker_counts.set_index("client_id"), on="client_id"
        )
        self.validated = self.validated.sort_values(
            ["user_sentence_count", "client_id"]
        )
        validated = self.validated.groupby("sentence").head(
            self.args.duplicate_sentence_count
        )

        validated = validated.sort_values(
            ["user_sentence_count", "client_id"], ascending=False
        )
        validated = validated.drop(columns="user_sentence_count")
        self.validated = self.validated.drop(columns="user_sentence_count")

        train = pd.DataFrame(columns=validated.columns)
        dev = pd.DataFrame(columns=validated.columns)
        test = pd.DataFrame(columns=validated.columns)

        train_size = dev_size = test_size = 0

        if len(validated) > 0:
            # Determine train, dev, and test sizes
            train_size, dev_size, test_size = self._calculate_data_set_sizes(
                len(validated)
            )
            # Split into train, dev, and test datasets
            continous_client_index, uniques = pd.factorize(validated["client_id"])
            validated["continous_client_index"] = continous_client_index

            for i in range(max(continous_client_index), -1, -1):
                if (
                    len(test) + len(validated[validated["continous_client_index"] == i])
                    <= test_size
                ):
                    test = pd.concat(
                        [test, validated[validated["continous_client_index"] == i]],
                        sort=False,
                    )
                elif (
                    len(dev) + len(validated[validated["continous_client_index"] == i])
                    <= dev_size
                ):
                    dev = pd.concat(
                        [dev, validated[validated["continous_client_index"] == i]],
                        sort=False,
                    )
                else:
                    train = pd.concat(
                        [train, validated[validated["continous_client_index"] == i]],
                        sort=False,
                    )

        self.train = train.drop(columns="continous_client_index", errors="ignore")
        self.dev = dev.drop(columns="continous_client_index", errors="ignore")
        self.test = test[:train_size].drop(
            columns="continous_client_index", errors="ignore"
        )

    def _calculate_data_set_sizes(self, total_size):
        # Find maximum size for the training data set in accord with sample theory
        train_size = total_size
        dev_size = test_size = 0
        for train_size in range(total_size, 0, -1):
            # calculated_sample_size = int(corporacreator.sample_size(train_size))
            calculated_sample_size = int(calc_sample_size(train_size))
            if 2 * calculated_sample_size + train_size <= total_size:
                dev_size = calculated_sample_size
                test_size = calculated_sample_size
                break
        return train_size, dev_size, test_size

    def save(self, directory):
        """Saves this :class:`corporacreator.Corpus` in `directory`.
        Args:
          directory (str): Directory into which this `corporacreator.Corpus` is saved.
        """
        directory = os.path.join(directory, self.locale)
        if not os.path.exists(directory):
            os.mkdir(directory)
        datasets = ["train", "dev", "test"]

        # _logger.debug("Saving %s corpora..." % self.locale)
        for dataset in datasets:
            self._save(directory, dataset)
        # _logger.debug("Saved %s corpora." % self.locale)

    def _save(self, directory, dataset):
        path = os.path.join(directory, dataset + ".tsv")

        dataframe = getattr(self, dataset)
        dataframe.to_csv(
            path,
            sep="\t",
            header=True,
            index=False,
            encoding="utf-8",
            escapechar="\\",
            quoting=csv.QUOTE_NONE,
        )


#
# PROCESS
# Handle one split creation, this is where calculations happen
#


# def corpora_creator_original(lc: str, val_path: str, dst_path: str, duplicate_sentences: int) -> bool:
def corpora_creator_original(val_path: str) -> bool:
    """Processes validated.tsv and create new train, dev, test splits"""
    dst_exppath: str = os.path.join(HERE, "experiments", DST_ALGO_DIR)
    results: list[bool] = []

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
        args = corporacreator.parse_args(
            ["-d", temp_path, "-f", val_path, "-s", str(DUPLICATE_SENTENCE_COUNT)]
        )
        corpus: LocalCorpus = LocalCorpus(args, lc, df_corpus)
        corpus.validated = df_corpus
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

    print(
        "=== Original Corpora Creator with -s 99 option for Common Voice Datasets ==="
    )

    # Paths
    experiments_path: str = os.path.join(HERE, "experiments")
    src_exppath: str = os.path.join(experiments_path, SRC_ALGO_DIR)
    dst_exppath: str = os.path.join(experiments_path, DST_ALGO_DIR)

    # Get total for progress display
    all_validated: "list[str]" = glob.glob(
        os.path.join(src_exppath, "**", "validated.tsv"), recursive=True
    )

    # For each corpus
    start_time: datetime = datetime.now()
    final_list: list[str] = []

    # clean unneeded/skipped
    if FORCE_CREATE:
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
    # print()   # extra line is for progress line

    # bar: progressbar.ProgressBar = progressbar.ProgressBar(max_value=g.SRC_COUNT, prefix="Dataset ")
    # bar.start()

    with mp.Pool(PROC_COUNT) as pool:
        results: list[bool] = pool.map(corpora_creator_original, final_list)

    for res in results:
        if res:
            g.processed_cnt += 1
        else:
            g.skipped_nodata += 1

    # end for
    # bar.finish()

    finish_time: datetime = datetime.now()
    process_timedelta: timedelta = finish_time - start_time
    process_seconds: float = process_timedelta.total_seconds()
    avg_seconds: float = -1
    avg_seconds_actual: float = -1
    if g.src_cnt > 0:
        avg_seconds = process_seconds / g.src_cnt
    if g.processed_cnt > 0:
        avg_seconds_actual = process_seconds / g.processed_cnt
    print("\n" + "-" * 80)
    print(
        f"Finished processing of {g.total_cnt} corpora in {str(process_timedelta)} secs"
    )
    print(
        f"Checked: {g.src_cnt}, Skipped (no-data): {g.skipped_nodata}, Actual: {g.processed_cnt}"
    )
    print(
        f'AVG in CHECKED {float("{:.3f}".format(avg_seconds))} secs, AVG in ACTUAL {float("{:.3f}".format(avg_seconds_actual))} secs, '
    )


if __name__ == "__main__":
    main()
