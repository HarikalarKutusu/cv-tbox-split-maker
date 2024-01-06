#!/usr/bin/env python3

###########################################################################
# algorithm-r1.py
#
# A work towards alternative proposal for current CorporaCreator implementation v2
#
# It uses validated.tsv to re-create train, dev, test splits
# using just shuffle and split, without any restrictions.
#
# Splits are 80-10-10% like others...
#
# This is created for being a baseline.
# It will use the whole validated recordings.
# As there are no restrictions, the resultant model will be randomşy biased.
#
# The script works on multiple CV versions and locales.
#
# The data is grouped as:
# experiments - Common Voice versions - locales - splits
#
# Use:
# python algorithm-r1.py
#
# This script is part of Common Voice ToolBox Package
#
# github: https://github.com/HarikalarKutusu/common-voice-diversity-check
# Copyright: (c) Bülent Özden, License: AGPL v3.0
###########################################################################

import os
import sys
import shutil
import glob
import csv

# from datetime import datetime
import pandas as pd

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

# Constants - TODO These should be arguments

SOURCE_EXPERIMENT_DIR: str = "s1"
DESTINATION_EXPERIMENT_DIR: str = "r1"
TRAIN_PERCENTAGE: float = 80.0
DEV_PERCENTAGE: float = 10.0
TEST_PERCENTAGE: float = 10.0

# Program parameters
VERBOSE: bool = True
FAIL_ON_NOT_FOUND: bool = True

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
# Handle one split creation, this is where calculations happen
#


def corpora_creator_v2(pth: str):
    """Processes validated.tsv and create new train, dev, test splits"""

    validated_path: str = os.path.join(pth, "validated.tsv")
    validated_df: pd.DataFrame = df_read(validated_path)

    # CALCULATE split sizes as record counts
    total_validated: int = validated_df.shape[0]

    test_target: int = int(TEST_PERCENTAGE / 100 * total_validated)
    dev_target: int = int(DEV_PERCENTAGE / 100 * total_validated)
    train_target: int = total_validated - dev_target - test_target

    if VERBOSE:
        print(
            f">>> Processing - {total_validated} validated records for RANDOM splits without restrictions. Targeting:"
        )
        print(f">>> TEST : {TEST_PERCENTAGE}% => {test_target} recs.")
        print(f">>> DEV  : {DEV_PERCENTAGE}% => {dev_target} recs.")
        print(f">>> TRAIN: {TRAIN_PERCENTAGE}% => {train_target} recs (remaining).")
        # print()

    # Randomize
    validated_df = validated_df.sample(frac=1).reset_index(drop=True)

    # Split
    test_df: pd.DataFrame = validated_df[:test_target]
    dev_df: pd.DataFrame = validated_df[test_target + 1 : test_target + dev_target]
    train_df: pd.DataFrame = validated_df[test_target + dev_target :]
    # Writeout results
    df_write(test_df, os.path.join(pth, "test.tsv"))
    df_write(dev_df, os.path.join(pth, "dev.tsv"))
    df_write(train_df, os.path.join(pth, "train.tsv"))
    # done
    return
    # sys.exit()


#
# Main loop for experiments-versions-locales
#


def main() -> None:
    print("=== New Corpora Creator Algorithm Proposal v3 for Common Voice Datasets ===")

    # Copy source experiment tree to destination experiment
    experiments_path: str = os.path.join(HERE, "experiments")
    src_exppath: str = os.path.join(experiments_path, SOURCE_EXPERIMENT_DIR)
    dst_exppath: str = os.path.join(experiments_path, DESTINATION_EXPERIMENT_DIR)
    shutil.copytree(src=src_exppath, dst=dst_exppath, dirs_exist_ok=True)
    # Remove old files
    to_delete: "list[str]" = []
    to_delete.extend(
        glob.glob(os.path.join(dst_exppath, "**", "train.tsv"), recursive=True)
    )
    to_delete.extend(
        glob.glob(os.path.join(dst_exppath, "**", "dev.tsv"), recursive=True)
    )
    to_delete.extend(
        glob.glob(os.path.join(dst_exppath, "**", "test.tsv"), recursive=True)
    )
    for fpath in to_delete:
        os.remove(fpath)

    # !!! from now on we will work on destination !!!

    exp_corpora_paths: "list[str]" = glob.glob(
        os.path.join(dst_exppath, "*"), recursive=False
    )

    # Get total for progress display
    all_validated: "list[str]" = glob.glob(
        os.path.join(dst_exppath, "**", "validated.tsv"), recursive=True
    )
    print(
        f"Re-splitting for {len(all_validated)} corpora... Wait for final structure is formed..."
    )
    print()  # extra line is for progress line

    # For each corpus
    cnt: int = 0  # counter of corpora done
    for corpus_path in exp_corpora_paths:
        exp_corpus_name: str = os.path.split(corpus_path)[-1]
        if VERBOSE:
            print(f"\n*** Processing Corpus: {exp_corpus_name} ***")
        # Now get the list of locales
        exp_corpus_locale_paths: "list[str]" = glob.glob(
            os.path.join(corpus_path, "*"), recursive=False
        )

        # For each locale
        for locale_path in exp_corpus_locale_paths:
            locale_name: str = os.path.split(locale_path)[-1]
            cnt += 1
            if VERBOSE:
                print(f"\n=== Processing Locale: {locale_name}\n")
            else:
                print("\033[F" + " " * 80)
                print(
                    f"\033[FProcessing {cnt}/{all_validated} => {exp_corpus_name} - {locale_name}"
                )
            # apply algorithm (splits are created there)
            corpora_creator_v2(pth=locale_path)
        # done locales in version
    # done version in versions


main()
