#!/usr/bin/env python3

###########################################################################
# proposal-n1.py
#
# Alternative proposal for current CorporaCreator implementation
#
# It uses validated.tsv to re-create train, dev, test splits
# using lowercased sentence recording frequencies.
#
# A sentence only lives in one split.
#
# This version does not take voices into account, a voice can live
# in any split.
#
# It practically uses the whole dataset.
#
# The script works on multiple CV versions and locales.
#
# The data is grouped as:
# experiments - Common Voice versions - locales - splits
#
# Use:
# python proposal-n1.py
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
DESTINATION_EXPERIMENT_DIR: str = "n1"
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

    def _check_intersection(
        df1: pd.DataFrame, df2: pd.DataFrame, df1_name: str, df2_name: str
    ) -> None:
        _intersect_df: pd.DataFrame = pd.merge(df1, df2, how="inner")
        if _intersect_df.shape[0] > 0:
            print(
                f"!!! ERROR IN ALGORITHM, SPLIT INTERSECTION FOUND - {df1_name} vs {df2_name}"
            )
            print(_intersect_df.head(999))
            sys.exit(1)
        elif VERBOSE:
            print(f"--- No intersection found - {df1_name} vs {df2_name} splits")

    validated_path: str = os.path.join(pth, "validated.tsv")
    validated_df: pd.DataFrame = df_read(validated_path)

    # add lowercase sentence column
    validated_df["sentence_lower"] = validated_df["sentence"].str.lower()

    # get unique lowercase sentences
    sentences_df: pd.DataFrame = (
        validated_df.groupby("sentence_lower").agg({"path": "count"}).reset_index()
    )  # get kst with count agg
    sentences_df.rename(
        columns={"path": "recorded_count"}, inplace=True
    )  # rename agg column
    sentences_df.sort_values(
        by="recorded_count", ascending=True, inplace=True
    )  # sort in ascending recorded count
    sentences_df["cumulative"] = sentences_df[
        "recorded_count"
    ].cumsum()  # add a cumulative sum for easy access
    sentences_df.reset_index()

    # CALCULATE split sizes as record counts
    total_validated: int = validated_df.shape[0]
    total_sentences: int = sentences_df.shape[0]

    test_target: int = int(TEST_PERCENTAGE / 100 * total_validated)
    dev_target: int = int(DEV_PERCENTAGE / 100 * total_validated)
    train_target: int = total_validated - dev_target - test_target
    if VERBOSE:
        print(
            f">>> Processing - {total_validated} validated records with {total_sentences} lower-case unique sentences."
        )
        print(
            f">>> Targeting - Train: {TRAIN_PERCENTAGE}%=>{train_target} recs, Dev: {DEV_PERCENTAGE}%=>{dev_target} recs, Test: {TEST_PERCENTAGE}%=>{test_target} recs "
        )
        # print()

    # Test
    _slice: pd.DataFrame = sentences_df[
        sentences_df["cumulative"].astype(int) <= test_target
    ]  # use cumulative column to get list of sentences to match the amount
    _sentences: list[str] = _slice["sentence_lower"].to_list()  # convert to list
    test_df: pd.DataFrame = validated_df[
        validated_df["sentence_lower"].isin(_sentences)
    ]  # select all validated records for that list
    test_df: pd.DataFrame = test_df.drop(
        columns=["sentence_lower"], errors="ignore"
    )  # drop temp columns
    test_voices: "list[str]" = test_df["client_id"].to_list()
    df_write(test_df, os.path.join(pth, "test.tsv"))  # output the result

    # Dev
    _slice: pd.DataFrame = sentences_df[
        (sentences_df["cumulative"].astype(int) > test_target)
        & (sentences_df["cumulative"].astype(int) <= test_target + dev_target)
    ]
    _sentences: list[str] = _slice["sentence_lower"].to_list()
    dev_df: pd.DataFrame = validated_df[validated_df["sentence_lower"].isin(_sentences)]
    dev_df: pd.DataFrame = dev_df.drop(columns=["sentence_lower"], errors="ignore")
    dev_voices: "list[str]" = dev_df["client_id"].to_list()
    df_write(dev_df, os.path.join(pth, "dev.tsv"))
    # check any possible intersection(s)
    _check_intersection(
        df1=dev_df, df2=test_df, df1_name="DEV Split", df2_name="TEST Split"
    )

    # Train
    _slice: pd.DataFrame = sentences_df[
        sentences_df["cumulative"].astype(int) > test_target + dev_target
    ]
    _sentences: list[str] = _slice["sentence_lower"].to_list()
    train_df: pd.DataFrame = validated_df[
        validated_df["sentence_lower"].isin(_sentences)
    ]
    train_df: pd.DataFrame = train_df.drop(columns=["sentence_lower"], errors="ignore")
    train_voices: "list[str]" = train_df["client_id"].to_list()
    df_write(train_df, os.path.join(pth, "train.tsv"))
    # check any possible intersection(s)
    _check_intersection(
        df1=train_df, df2=test_df, df1_name="TRAIN Split", df2_name="TEST Split"
    )
    _check_intersection(
        df1=train_df, df2=dev_df, df1_name="TRAIN Split", df2_name="DEV Split"
    )

    if VERBOSE:
        print(
            f">>> Result splits - Train: {train_df.shape[0]} recs, Dev: {dev_df.shape[0]} recs, Test: {test_df.shape[0]} recs "
        )
        print("--- Voice overlaps:")
        print(
            f"--- TRAIN has voices already in DEV: {len(set(train_voices).intersection(dev_voices))}"
        )
        print(
            f"--- TRAIN has voices already in TEST: {len(set(train_voices).intersection(test_voices))}"
        )
        print(
            f"--- DEV has voices already in TEST: {len(set(dev_voices).intersection(test_voices))}"
        )

    # done


#
# Main loop for versions-locales
#


def main() -> None:
    """A New Corpora Creator Algorithm for Common Voice Datasets"""
    print("=== A New Corpora Creator Algorithm for Common Voice Datasets ===")

    # Copy source experiment tree to destination experiment
    experiments_path: str = os.path.join(HERE, "experiments")
    src_exppath: str = os.path.join(experiments_path, SOURCE_EXPERIMENT_DIR)
    dst_exppath: str = os.path.join(experiments_path, DESTINATION_EXPERIMENT_DIR)
    shutil.copytree(src=src_exppath, dst=dst_exppath, dirs_exist_ok=True)
    # !!! from now on we will work on destination !!!
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

    exp_corpora_paths: "list[str]" = glob.glob(
        os.path.join(dst_exppath, "*"), recursive=False
    )

    # Get total for progress display
    all_validated: "list[str]" = glob.glob(
        os.path.join(dst_exppath, "**", "validated.tsv"), recursive=True
    )
    print(f"Re-splitting for {len(all_validated)} corpora...")
    print()  # extra line is for progress line

    # For each corpus
    cnt: int = 0  # counter of corpora done
    for corpus_path in exp_corpora_paths:
        exp_corpus_name: str = os.path.split(corpus_path)[-1]
        if VERBOSE:
            print(f"== Processing Corpus: {exp_corpus_name}")
        # Now get the list of locales
        exp_corpus_locale_paths: "list[str]" = glob.glob(
            os.path.join(corpus_path, "*"), recursive=False
        )

        # For each locale
        for locale_path in exp_corpus_locale_paths:
            locale_name: str = os.path.split(locale_path)[-1]
            cnt += 1
            if VERBOSE:
                print(f"=== Processing Locale: {locale_name}")
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
