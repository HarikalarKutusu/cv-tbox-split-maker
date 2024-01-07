#!/usr/bin/env python3
"""
cv-tbox Diversity Check / Split Maker
Alternative splitting algorithm which uses all recordings (v1) more optimized for Whisper usage
"""
###########################################################################
# algorithm-vw.py
#
# Alternative proposal for whisper fine-tuning.
# It uses same algorithm as v1, but
# - with 90%-5%-5% splits
# - user limits 25%/25%/50%
#
# Use:
# python algorithm-w1.py
#
# This script is part of Common Voice ToolBox Package
#
# github: https://github.com/HarikalarKutusu/common-voice-diversity-check
# Copyright: (c) Bülent Özden, License: AGPL v3.0
###########################################################################

# Standard Lib
import os
import sys
import glob
import multiprocessing as mp
from multiprocessing.pool import AsyncResult
from datetime import datetime

# External dependencies
import pandas as pd
import psutil
from tqdm import tqdm

# Module
from languages import WHISPER_LANGUAGES
from typedef import AlgorithmSpecs, AlgorithmResults, Globals
from lib import df_read, df_write, final_report
import conf

#
# CONST
#
SAMPLE_SIZE_THRESHOLD: int = 150000
MIN_VALIDATED_THRESHOLD = 2000

#
# Globals
#
HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

PROC_COUNT: int = psutil.cpu_count(logical=True)  # Full usage

#
# Globals
#
g = Globals()
aspecs = AlgorithmSpecs(
    src_algo_dir="s1",
    dst_algo_dir="vw",
    train_percentage=90.0,
    dev_percentage=5.0,
    test_percentage=5.0,
    max_test_user=0.50,
    max_dev_user=0.25,
)


#
# Handle one split creation, this is where calculations happen
#
def algorithm_vw(val_path: str) -> AlgorithmResults:
    """Processes validated.tsv and create new train, dev, test splits"""

    results: AlgorithmResults = AlgorithmResults()

    src_corpus_dir: str = os.path.split(val_path)[0]
    lc: str = os.path.split(src_corpus_dir)[1]
    ver: str = os.path.split(os.path.split(src_corpus_dir)[0])[1]
    dst_path: str = os.path.join(HERE, "experiments", aspecs.dst_algo_dir, ver, lc)
    results.lc = lc
    results.ver = ver
    # results.ver = ver.replace("cv-corpus-", "")
    if conf.VERBOSE:
        print(f"Executing {ver} - {lc}", flush=True)

    validated_df: pd.DataFrame = df_read(val_path)

    total_validated: int = validated_df.shape[0]
    if total_validated == 0:
        results.tiny = 1
        results.skipped_nodata = 1
        return results
    if total_validated < MIN_VALIDATED_THRESHOLD:
        results.tiny = 1
        results.skipped_small = 1
        return results

    os.makedirs(dst_path, exist_ok=True)

    # add lowercase sentence column
    validated_df["sentence_lower"] = validated_df["sentence"].str.lower()
    # add an enumaration column for client_id's
    validated_df["v_enum"], _ = pd.factorize(validated_df["client_id"])
    # add an enumaration column for client_id's
    validated_df["s_enum"], _ = pd.factorize(validated_df["sentence_lower"])

    # Try with unique voices

    # get list with count agg
    voices_df: pd.DataFrame = (
        validated_df.groupby("v_enum")
        .agg({"path": "count", "s_enum": "count"})
        .reset_index()
    )
    # rename agg column
    voices_df.rename(
        columns={"path": "recorded_count", "client_id": "sentence_count"}, inplace=True
    )
    # sort in ascending recorded count
    voices_df.sort_values(by=["recorded_count", "v_enum"], ascending=True, inplace=True)
    # add a cumulative sum for easy access
    voices_df["cumulative_recordings"] = voices_df["recorded_count"].cumsum()
    voices_df.reset_index()

    # CALCULATE split sizes as record counts
    total_validated: int = validated_df.shape[0]
    # total_sentences: int = validated_df["s_enum"].max()
    total_voices: int = voices_df.shape[0]

    test_voice_max: int = int(aspecs.max_test_user * total_voices)
    dev_voice_max: int = int(aspecs.max_dev_user * total_voices)

    # Adaptive part - if population size >= 150.000, then use sample size calculation, else use percentages given

    #
    # Tag Large & Madium, tiny is out of question
    #
    if total_validated >= SAMPLE_SIZE_THRESHOLD:
        results.large = 1
    else:
        results.medium = 1

    test_target: int = int(aspecs.test_percentage / 100 * total_validated)
    dev_target: int = int(aspecs.dev_percentage / 100 * total_validated)
    # train_target: int = total_validated - dev_target - test_target

    #
    # STEP-1 : First run to predict slices for splits
    #

    # TEST

    # Test use cumsum to get them quickly
    # use cumulative column to get list of user recordings to match the amount
    test_slice: pd.DataFrame = voices_df[
        voices_df["cumulative_recordings"].astype(int) <= test_target
    ].reset_index(drop=True)
    # If sliced records are more than test_voice_max we need to re-slice
    if test_slice.shape[0] > test_voice_max:
        # This time get the first N voices
        test_slice: pd.DataFrame = voices_df[0:test_voice_max]
    actual_test_target: int = int(test_slice["cumulative_recordings"].iat[-1])

    # DEV

    voices_df: pd.DataFrame = voices_df[test_slice.shape[0] :]
    # use cumulative column to get list of user recordings to match the amount
    dev_slice: pd.DataFrame = voices_df[
        voices_df["cumulative_recordings"].astype(int)
        <= actual_test_target + dev_target
    ].reset_index(drop=True)
    if dev_slice.shape[0] > dev_voice_max:
        # Get based on number of voices
        dev_slice: pd.DataFrame = voices_df[0:dev_voice_max]
    actual_dev_target: int = int(dev_slice["cumulative_recordings"].iat[-1])

    # TRAIN
    # Get the rest
    train_slice: pd.DataFrame = voices_df[
        voices_df["cumulative_recordings"].astype(int) > actual_dev_target
    ].reset_index(drop=True)

    #
    # STEP-2 : Now swap TEST's high end voices & DEV's high voices end with low end of TRAIN in order to fulfill the target split size.
    #
    delta_test_df: pd.DataFrame = pd.DataFrame(columns=voices_df.columns)
    delta_dev_df: pd.DataFrame = pd.DataFrame(columns=voices_df.columns)
    delta_train_df: pd.DataFrame = pd.DataFrame(columns=voices_df.columns)

    # Handle TEST-TRAIN
    test_missing: int = test_target - actual_test_target  # calc how much missing
    # do it only missing & possible
    # if test_missing > 0 and test_slice.shape[0] > 5 and train_slice.shape[0] > 5:
    if test_missing > 0 and test_slice.shape[0] > 5 and train_slice.shape[0] > 5:
        inx: int = -1
        delta_test: int = 0
        delta_train: int = 0
        while (delta_train - delta_test) < test_missing:
            inx += 1
            # start from lowest to higher
            delta_train += int(train_slice["recorded_count"].iat[inx])
            # start from highest to lower
            delta_test += int(test_slice["recorded_count"].iat[-inx])
            # print('...step...', inx, delta_train, delta_test)

        # Get the tail to move to train (will happen later)
        delta_test_df: pd.DataFrame = test_slice[-inx - 1 :]
        # Get the head to move to test
        delta_train_df: pd.DataFrame = train_slice[: inx + 1]
        # To head of test, append head of train
        test_slice: pd.DataFrame = pd.concat([test_slice[: -inx - 1], delta_train_df])
        # Make a smaller train by removing the moved head
        train_slice: pd.DataFrame = train_slice[inx + 1 :]

    # Handle DEV-TRAIN
    # calc how much missing
    dev_missing: int = dev_target - (actual_dev_target - actual_test_target)
    # do it only missing & possible
    if dev_missing > 0 and dev_slice.shape[0] > 5 and train_slice.shape[0] > 5:
        inx: int = -1
        delta_dev: int = 0
        delta_train: int = 0
        while (delta_train - delta_dev) < dev_missing:
            inx += 1
            # start from lowest to higher
            delta_train += int(train_slice["recorded_count"].iat[inx])
            # start from highest to lower
            delta_dev += int(dev_slice["recorded_count"].iat[-inx])
            # print('...step...', inx, delta_train, delta_dev)

        delta_dev_df: pd.DataFrame = dev_slice[-inx - 1 :]
        delta_train_df: pd.DataFrame = train_slice[: inx + 1]
        dev_slice: pd.DataFrame = pd.concat([dev_slice[: -inx - 1], delta_train_df])
        train_slice: pd.DataFrame = train_slice[inx + 1 :]

    # Here we have CUT OUT train, now we maybe add the swapped parts
    if delta_dev_df.shape[0] > 0:
        train_slice: pd.DataFrame = pd.concat([delta_dev_df.loc[:], train_slice])
    if delta_test_df.shape[0] > 0:
        train_slice: pd.DataFrame = pd.concat([delta_test_df.loc[:], train_slice])

    #
    # STEP-3 : Now, build the actual split DF's
    #

    # get a list of unique voice enum in test set
    test_voices: "list[str]" = test_slice["v_enum"].unique().tolist()
    # select all validated records for that list
    test_df: pd.DataFrame = validated_df[validated_df["v_enum"].isin(test_voices)]

    # Do it again for DEV
    # get a list of unique voice enum in dev set
    dev_voices: "list[str]" = dev_slice["v_enum"].unique().tolist()
    # get a list of unique voice enum in dev set
    # total_dev_voices: int = len(dev_voices)
    # select all validated records for that list
    dev_df: pd.DataFrame = validated_df[validated_df["v_enum"].isin(dev_voices)]

    # Rest will be in TRAIN
    train_voices: "list[str]" = train_slice["v_enum"].unique().tolist()
    # get a list of unique voice enum in test set
    # total_train_voices: int = len(train_voices)
    # get remaining directly from validated
    train_df: pd.DataFrame = validated_df[validated_df["v_enum"].isin(train_voices)]

    # Remove extra columns
    test_df: pd.DataFrame = test_df.drop(
        columns=["sentence_lower", "v_enum", "s_enum"], errors="ignore"
    )
    dev_df: pd.DataFrame = dev_df.drop(
        columns=["sentence_lower", "v_enum", "s_enum"], errors="ignore"
    )
    train_df: pd.DataFrame = train_df.drop(
        columns=["sentence_lower", "v_enum", "s_enum"], errors="ignore"
    )

    # Writeout results
    df_write(test_df, os.path.join(dst_path, "test.tsv"))
    df_write(dev_df, os.path.join(dst_path, "dev.tsv"))
    df_write(train_df, os.path.join(dst_path, "train.tsv"))

    results.processed = 1
    return results


#
# Main loop for experiments-versions-locales
#
def main() -> None:
    """Main function which calls multiprocessing"""

    #
    # Callback
    #

    def pool_callback(res: AlgorithmResults) -> None:
        """Callback to append results and increment bar"""
        # print(f"Finished {res.lc}")
        pbar.update()
        try:
            g.processed_cnt += res.processed
            g.skipped_small += res.skipped_small
            g.skipped_nodata += res.skipped_nodata
            g.tiny_dataset_cnt += res.tiny
            g.medium_dataset_cnt += res.medium
            g.large_dataset_cnt += res.large
        except ValueError as e:
            print(f"Failed for {result}:", e)

    #
    # Main
    #
    print("=== vw algorithm for whisper fine-tuning for Common Voice Datasets ===")

    g.start_time = datetime.now()

    # Copy source experiment tree to destination experiment
    src_exppath: str = os.path.join(HERE, "experiments", aspecs.src_algo_dir)
    dst_exppath: str = os.path.join(HERE, "experiments", aspecs.dst_algo_dir)

    # Get total for progress display
    all_validated: "list[str]" = glob.glob(
        os.path.join(src_exppath, "**", "validated.tsv"), recursive=True
    )

    # clean unneeded/skipped
    final_list: list[str] = []

    for val_path in all_validated:
        src_corpus_dir: str = os.path.split(val_path)[0]
        lc: str = os.path.split(src_corpus_dir)[1]
        ver: str = os.path.split(os.path.split(src_corpus_dir)[0])[1]
        dst_corpus_dir: str = os.path.join(dst_exppath, ver, lc)

        if lc not in WHISPER_LANGUAGES:
            # We need it to be among whisper languages
            g.skipped_nosupport += 1
        elif not conf.FORCE_CREATE and os.path.isfile(
            # if not force create, we skip if it exists
            os.path.join(dst_corpus_dir, "train.tsv")
        ):
            # Already there and is not forced to recreate, so skip
            g.skipped_exists += 1
        else:
            final_list.append(val_path)

    g.total_cnt = len(all_validated)
    g.src_cnt = len(final_list)

    print(
        f"Re-splitting for {g.src_cnt} out of {g.total_cnt} corpora in {PROC_COUNT} processes."
    )
    print(f"Skipping Existing: {g.skipped_exists} & Not Supported: {g.skipped_nosupport}")

    chunk_size: int = g.src_cnt // PROC_COUNT + 0 if g.src_cnt % PROC_COUNT == 0 else 1

    with mp.Pool(PROC_COUNT) as pool:
        with tqdm(total=g.src_cnt) as pbar:
            for result in pool.imap_unordered(
                algorithm_vw, final_list, chunksize=chunk_size
            ):
                pool_callback(result)


    final_report(g)


if __name__ == "__main__":
    main()
