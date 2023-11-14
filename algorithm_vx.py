#!/usr/bin/env python3
"""
cv-tbox Diversity Check / Split Maker
A version of v1 algorithm which produces train & dev, where external dataset like Fleurs is used as test
"""
###########################################################################
# proposal-vx.py
#
# Alternative proposal for whisper fine-tuning.
# It uses same algorithm as v1, but
# - with 90%-5%-5% splits
# - user limits 50%/20%/30%
#
# Use:
# python algorithm-w1.py
#
# This script is part of Common Voice ToolBox Package
#
# [github]
# [copyright]
###########################################################################

# Standard Lib
import os
import sys
import glob
import csv
from datetime import datetime, timedelta

# External dependencies
import pandas as pd

# Module
from languages import LANGUAGES_ALLOWED
from lib import df_read, df_write
import conf

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

#
# Constants - TODO These should be arguments
#

# Real Constants
SAMPLE_SIZE_THRESHOLD: int = 150000
MIN_VALIDATED_THRESHOLD = 1000

# Directories
SRC_ALGO_DIR: str = "s1"
DST_ALGO_DIR: str = "vx"


TRAIN_PERCENTAGE: float = 95.0
DEV_PERCENTAGE: float = 5.0

MAX_DEV_USER: float = 0.50

cnt_processed: int = 0  # counter of corpora processed
cnt_skipped: int = 0  # count of corpora skipped
num_total: int = 0  # total number of datasets


#
# Handle one split creation, this is where calculations happen
#


def corpora_creator_vx(val_path: str, dst_path: str):
    """Processes validated.tsv and create new train, dev splits"""

    global cnt_skipped

    # #
    # # Sample Size Calculation, taken from CorporaCreaotr repo (statistics.py)
    # #
    # def calc_sample_size(population_size: int) -> float:
    #     """Calculates the sample size.

    #     Calculates the sample size required to draw from a population size `population_size`
    #     with a confidence level of 99% and a margin of error of 1%.

    #     Args:
    #     population_size (int): The population size to draw from.
    #     """
    #     margin_of_error: float = 0.01
    #     fraction_picking: float = 0.50
    #     z_score: float = 2.58  # Corresponds to confidence level 99%
    #     numerator: float = (
    #         z_score**2 * fraction_picking * (1 - fraction_picking)
    #     ) / (margin_of_error**2)
    #     denominator: float = 1 + (
    #         z_score**2 * fraction_picking * (1 - fraction_picking)
    #     ) / (margin_of_error**2 * population_size)
    #     return numerator / denominator

    validated_df: pd.DataFrame = df_read(val_path)

    # SKIP SMALL DATASETS
    total_validated: int = validated_df.shape[0]
    if total_validated < MIN_VALIDATED_THRESHOLD:
        cnt_skipped += 1
        print(
            "!!! SKIP SMALL DATASET IN: [",
            " | ".join(val_path.split(os.sep)[-3:-1]),
            "] with records:",
            total_validated,
        )
        return

    os.makedirs(dst_path, exist_ok=True)

    # add lowercase sentence column
    validated_df["sentence_lower"] = validated_df["sentence"].str.lower()
    validated_df["v_enum"], v_unique = pd.factorize(
        validated_df["client_id"]
    )  # add an enumaration column for client_id's
    validated_df["s_enum"], s_unique = pd.factorize(
        validated_df["sentence_lower"]
    )  # add an enumaration column for client_id's

    # Try with unique voices
    voices_df: pd.DataFrame = (
        validated_df.groupby("v_enum")
        .agg({"path": "count", "s_enum": "count"})
        .reset_index()
    )  # get list with count agg
    voices_df.rename(
        columns={"path": "recorded_count", "client_id": "sentence_count"}, inplace=True
    )  # rename agg column
    voices_df.sort_values(
        by=["recorded_count", "v_enum"], ascending=True, inplace=True
    )  # sort in ascending recorded count
    voices_df["cumulative_recordings"] = voices_df[
        "recorded_count"
    ].cumsum()  # add a cumulative sum for easy access
    # sentences_df['s_enum'], s_unique = pd.factorize(sentences_df['sentence_lower'])                                     # add an enumaration column for sentences (lower)
    voices_df.reset_index()

    # CALCULATE split sizes as record counts
    total_sentences: int = validated_df["s_enum"].max()
    total_voices: int = voices_df.shape[0]

    dev_voice_max: int = int(MAX_DEV_USER * total_voices)

    # Adaptive part - if population size >= 150.000, then use sample size calculation, else use percentages given

    # DO NOT USE ADAPTIVE PART
    # Use given percentages!
    dev_target: int = int(DEV_PERCENTAGE / 100 * total_validated)
    train_target: int = total_validated - dev_target
    if conf.VERBOSE:
        print(
            f">>> Processing - {total_validated} validated records with {total_sentences} lower-case unique sentences from {total_voices} voices. Targeting:"
        )
        print(
            f">>> DEV  : {DEV_PERCENTAGE}% => {dev_target} recs OR max {dev_voice_max} voices"
        )
        print(
            f">>> TRAIN: {TRAIN_PERCENTAGE}% => {train_target} recs OR remaining from TEST & DEV"
        )
        # print()

    if total_validated < 100 or total_voices < 10:
        print("!!! TOO LOW ON RESOURCES, SPLITTING RANDOMLY !!!")
        # Remove extra columns
        dev_df: pd.DataFrame = validated_df[:dev_target]
        train_df: pd.DataFrame = validated_df[dev_target:]
        # Writeout results
        df_write(dev_df, os.path.join(dst_path, "dev.tsv"))
        df_write(train_df, os.path.join(dst_path, "train.tsv"))
        return

    #
    # STEP-1 : First run to predict slices for splits
    #

    # DEV

    # Use cumsum to get them quickly
    dev_slice: pd.DataFrame = voices_df[
        voices_df["cumulative_recordings"].astype(int) <= dev_target
    ].reset_index(
        drop=True
    )  # use cumulative column to get list of user recordings to match the amount
    # If sliced records are more than dev_voice_max we need to re-slice
    if dev_slice.shape[0] > dev_voice_max:
        if conf.VERBOSE:
            print("dev-Re-sliced because max voices exceeded")
        dev_slice: pd.DataFrame = voices_df[
            0:dev_voice_max
        ]  # This time get the first N voices
    actual_dev_target: int = int(dev_slice["cumulative_recordings"].iat[-1])

    # TRAIN

    train_slice: pd.DataFrame = voices_df[
        voices_df["cumulative_recordings"].astype(int) > actual_dev_target
    ].reset_index(
        drop=True
    )  # Get the rest

    # if conf.VERBOSE:
    #     print(f'VOICES: TEST={test_slice.shape[0]}/{test_voice_max}   DEV={dev_slice.shape[0]}/{dev_voice_max}   TRAIN={train_slice.shape[0]} TOTAL={train_slice.shape[0] + dev_slice.shape[0] + test_slice.shape[0]}/{total_voices}')
    #     print(f'ACTUAL: TEST={actual_test_target}/{test_target} DEV={actual_dev_target - actual_test_target}/{dev_target}')

    #
    # STEP-2 : Now swap TEST's high end voices & DEV's high voices end with low end of TRAIN in order to fulfill the target split size.
    #

    # print('SLICES:', test_slice.shape, dev_slice.shape, train_slice.shape)

    delta_dev_df: pd.DataFrame = pd.DataFrame(columns=voices_df.columns)
    delta_train_df: pd.DataFrame = pd.DataFrame(columns=voices_df.columns)

    # Handle DEV-TRAIN
    dev_missing: int = dev_target - actual_dev_target  # calc how much missing
    # print('Missing recs in DEV=', dev_missing)
    if (
        dev_missing > 0 and dev_slice.shape[0] > 5 and train_slice.shape[0] > 5
    ):  # do it only missing & possible
        inx: int = -1
        delta_dev: int = 0
        delta_train: int = 0
        limit_reached: bool = False
        while (delta_train - delta_dev) < dev_missing and not limit_reached:
            inx += 1
            try:
                delta_train += int(
                    train_slice["recorded_count"].iat[inx]
                )  # start from lowest to higher
                delta_dev += int(
                    dev_slice["recorded_count"].iat[-inx]
                )  # start from highest to lower
            except:
                limit_reached = True
            # print('...step...', inx, delta_train, delta_dev)
        # here we know
        if conf.VERBOSE:
            print(
                f"SWAP DEV-TRAIN {inx+1} VOICES, FOR {delta_train - delta_dev} RECORDINGS TO FILL {dev_missing} MISSING RECS IN DEV SPLIT"
            )
        # print('OLD TRAIN:\n', train_slice.head(inx+2))
        # print('OLD DEV:\n', dev_slice.tail(inx+2))
        delta_dev_df: pd.DataFrame = dev_slice[-inx - 1 :]
        delta_train_df: pd.DataFrame = train_slice[: inx + 1]
        dev_slice: pd.DataFrame = pd.concat([dev_slice[: -inx - 1], delta_train_df])
        train_slice: pd.DataFrame = train_slice[inx + 1 :]
        # print('DDEV:\n', delta_dev_df.head(inx+2))
        # print('DTRAIN:\n', delta_train_df.head(inx+2))
        # print('NEW TRAIN:\n', train_slice.head(inx+2))
        # print('NEW DEV:\n', dev_slice.tail(inx+2))
        # print('DELTAS:', delta_dev_df.shape, delta_train_df.shape)
        # print('SLICES:', test_slice.shape, dev_slice.shape, train_slice.shape)

    # Here we have CUT OUT train, now we maybe add the swapped parts
    # print('DELTAS:', delta_test_df.shape, delta_dev_df.shape, delta_train_df.shape)
    if delta_dev_df.shape[0] > 0:
        train_slice: pd.DataFrame = pd.concat([delta_dev_df.loc[:], train_slice])

    # print('SLICES:', test_slice.shape, dev_slice.shape, train_slice.shape)
    #
    # STEP-3 : Now, build the actual split DF's
    #

    # Do it again for DEV
    dev_voices: "list[str]" = (
        dev_slice["v_enum"].unique().tolist()
    )  # get a list of unique voice enum in dev set
    total_dev_voices: int = len(
        dev_voices
    )  # get a list of unique voice enum in dev set
    dev_df: pd.DataFrame = validated_df[
        validated_df["v_enum"].isin(dev_voices)
    ]  # select all validated records for that list
    total_dev_sentences: int = len(
        dev_df["s_enum"].unique().tolist()
    )  # get a list of unique voice enum in test set
    dev_recs: int = dev_df.shape[0]
    # if conf.VERBOSE:
    #     print(f'--> DEV   split now has {dev_recs} records with {total_dev_sentences} distinct sentences from {total_dev_voices} distinct voices (from {total_voices})')

    # Rest will be in TRAIN
    train_voices: "list[str]" = train_slice["v_enum"].unique().tolist()
    total_train_voices: int = len(
        train_voices
    )  # get a list of unique voice enum in test set
    train_df: pd.DataFrame = validated_df[
        validated_df["v_enum"].isin(train_voices)
    ]  # get remaining directly from validated
    total_train_sentences: int = len(
        train_df["s_enum"].unique().tolist()
    )  # get a list of unique voice enum in test set
    train_recs: int = train_df.shape[0]
    if conf.VERBOSE:
        # print(f'--> TRAIN split now has {train_recs} records with {total_train_sentences} distinct sentences from {total_train_voices} distinct voices (from {total_voices})')
        tot_v: int = total_train_voices + total_dev_voices
        tot_r: int = train_recs + dev_recs
        print(
            f'--> RESULT VOICES  (TRAIN + DEV) = {total_train_voices} ({float("{:.2f}".format(100 * total_train_voices/tot_v))}%) + '
            + f'{total_dev_voices} ({float("{:.2f}".format(100 * total_dev_voices/tot_v))}%) + '
            + f"{tot_v} (expecting {total_voices})"
        )
        print(
            f'--> RESULT RECORDS (TRAIN + DEV) = {train_recs} ({float("{:.2f}".format(100 * train_recs/tot_r))}%) + '
            + f'{dev_recs} ({float("{:.2f}".format(100 * dev_recs/tot_r))}%) + '
            + f"{train_recs + dev_recs} (expecting {total_validated})"
        )

    # Remove extra columns
    dev_df: pd.DataFrame = dev_df.drop(
        columns=["sentence_lower", "v_enum", "s_enum"], errors="ignore"
    )
    train_df: pd.DataFrame = train_df.drop(
        columns=["sentence_lower", "v_enum", "s_enum"], errors="ignore"
    )

    # Writeout results
    df_write(dev_df, os.path.join(dst_path, "dev.tsv"))
    df_write(train_df, os.path.join(dst_path, "train.tsv"))

    # done
    # sys.exit()


#
# Main loop for experiments-versions-locales
#


def main() -> None:
    """vx is a custom v1 algorithm to create only train & dev splits to be used against external test"""
    global cnt_skipped
    global cnt_processed
    global num_total

    print("=== vx algorithm to be used with external test data ===")

    # Copy source experiment tree to destination experiment
    experiments_path: str = os.path.join(HERE, "experiments")
    src_exppath: str = os.path.join(experiments_path, SRC_ALGO_DIR)
    dst_exppath: str = os.path.join(experiments_path, DST_ALGO_DIR)
    # shutil.copytree(src=src_exppath, dst=dst_exppath, dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.tsv'))

    # !!! from now on we will work on destination !!!

    # src_corpora_paths: "list[str]" = glob.glob(os.path.join(src_exppath, '*'), recursive=False)
    # dst_corpora_paths: "list[str]" = glob.glob(os.path.join(dst_exppath, '*'), recursive=False)

    # Get total for progress display
    all_validated: "list[str]" = glob.glob(
        os.path.join(src_exppath, "**", "validated.tsv"), recursive=True
    )
    print(
        f"Re-splitting for {len(all_validated)} corpora... Wait for final structure is formed..."
    )
    print()  # extra line is for progress line

    # For each corpus
    start_time: datetime = datetime.now()
    num_total = len(all_validated)

    for val_path in all_validated:
        src_corpus_dir: str = os.path.split(val_path)[0]
        lc: str = os.path.split(src_corpus_dir)[1]
        ver: str = os.path.split(os.path.split(src_corpus_dir)[0])[1]
        dst_corpus_dir: str = os.path.join(dst_exppath, ver, lc)

        cnt_processed += 1
        if conf.VERBOSE:
            print(f"\n=== Processing {cnt_processed}/{num_total} => {ver} - {lc}\n")
        else:
            print("\033[F" + " " * 80)
            print(f"\033[FProcessing {cnt_processed}/{num_total} => {ver} - {lc}")

        if lc not in LANGUAGES_ALLOWED:
            # We need it to be among allowed languages
            print(f"!!! SKIP UNSUPPORTED: [{lc}]")
            cnt_skipped += 1
        elif not conf.FORCE_CREATE and os.path.isfile(
            os.path.join(dst_corpus_dir, "train.tsv")
        ):
            # Already there and is not forced to recreate, so skip
            cnt_skipped += 1
        else:
            corpora_creator_vx(val_path=val_path, dst_path=dst_corpus_dir)
            print()

    finish_time: datetime = datetime.now()
    process_timedelta: timedelta = finish_time - start_time
    process_seconds: float = process_timedelta.total_seconds()
    avg_seconds: float = process_seconds / num_total
    cnt_new: int = cnt_processed - cnt_skipped
    avg_seconds_new: float = -1
    if cnt_new > 0:
        avg_seconds_new: float = process_seconds / cnt_new
    print("\n" + "-" * 80)
    print(f"Processed: {cnt_new}, Skipped: {cnt_skipped}, New: {cnt_new}")
    print(
        f'Duration: {str(process_timedelta)}s, avg duration {float("{:.3f}".format(avg_seconds))}s'
    )
    if cnt_new > 0:
        print(
            f'Avg. time new split creation: {float("{:.3f}".format(avg_seconds_new))}'
        )


if __name__ == "__main__":
    main()
