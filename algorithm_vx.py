#!/usr/bin/env python3
"""
cv-tbox Diversity Check / Split Maker
A version of v1 algorithm which produces train & dev, where external dataset like Fleurs is used as test
"""
###########################################################################
# proposal-vx.py
#
# Alternative proposal for whisper fine-tuning and testing against an external dataset like Fleurs or Voxpopuli.
# It uses same algorithm as v1, but
# - with 90%-5%-0% splits
# - user limits 50%/50%/0%
#
# So, there is no test split.
#
# Use:
# python algorithm-w1.py
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
from multiprocessing.pool import AsyncResult
from datetime import datetime

# External dependencies
import pandas as pd
import psutil
from tqdm import tqdm

# Module
from languages import LANGUAGES_ALLOWED
from typedef import AlgorithmSpecs, AlgorithmResults, Globals
from lib import df_read, df_write, final_report, remove_deleted_users
import conf

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
    dst_algo_dir="vx",
    train_percentage=95.0,
    dev_percentage=5.0,
    test_percentage=0.0,
    max_test_user=0.0,
    max_dev_user=0.50,
)


#
# Handle one split creation, this is where calculations happen
#
def algorithm_vx(val_path: str) -> AlgorithmResults:
    """Processes validated.tsv and create new train, dev splits"""

    results: AlgorithmResults = AlgorithmResults()

    src_corpus_dir: str = os.path.split(val_path)[0]
    lc: str = os.path.split(src_corpus_dir)[1]
    ver: str = os.path.split(os.path.split(src_corpus_dir)[0])[1]
    dst_path: str = os.path.join(
        conf.SM_DATA_DIR, "experiments", aspecs.dst_algo_dir, ver, lc
    )
    results.lc = lc
    results.ver = ver
    # results.ver = ver.replace("cv-corpus-", "")
    if conf.VERBOSE:
        print(f"Executing {ver} - {lc}", flush=True)

    validated_df: pd.DataFrame = df_read(val_path)
    num_original: int = validated_df.shape[0]

    # Remove users who requested data deletion
    validated_df = remove_deleted_users(validated_df)
    total_validated: int = validated_df.shape[0]
    if num_original != total_validated and conf.VERBOSE:
        print(
            f"\nUSER RECORDS DELETED FROM VALIDATED {ver}-{lc} = {num_original - validated_df.shape[0]}"
        )

    if total_validated == 0:
        results.tiny = 1
        results.skipped_nodata = 1
        return results
    if total_validated < conf.MIN_VALIDATED_THRESHOLD:
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

    #
    # Tag Large & Madium, tiny is out of question
    #
    if total_validated >= conf.SAMPLE_SIZE_THRESHOLD:
        results.large = 1
    else:
        results.medium = 1

    # DO NOT USE ADAPTIVE PART
    # Use given percentages!
    dev_target: int = int(aspecs.dev_percentage / 100 * total_validated)
    # train_target: int = total_validated - dev_target

    #
    # STEP-1 : First run to predict slices for splits
    #

    # DEV

    # Use cumsum to get them quickly
    # use cumulative column to get list of user recordings to match the amount
    dev_slice: pd.DataFrame = voices_df[
        voices_df["cumulative_recordings"].astype(int) <= dev_target
    ].reset_index(drop=True)
    # If sliced records are more than dev_voice_max we need to re-slice
    # This time get the first N voices
    if dev_slice.shape[0] > dev_voice_max:
        dev_slice: pd.DataFrame = voices_df[0:dev_voice_max]
    actual_dev_target: int = int(dev_slice["cumulative_recordings"].iat[-1])

    # TRAIN
    # Get the rest
    train_slice: pd.DataFrame = voices_df[
        voices_df["cumulative_recordings"].astype(int) > actual_dev_target
    ].reset_index(drop=True)

    #
    # STEP-2 : Now swap TEST's high end voices & DEV's high voices end with low end of TRAIN
    # in order to fulfill the target split size.
    #

    delta_dev_df: pd.DataFrame = pd.DataFrame(columns=voices_df.columns)
    delta_train_df: pd.DataFrame = pd.DataFrame(columns=voices_df.columns)

    # Handle DEV-TRAIN
    # calc how much missing
    dev_missing: int = dev_target - actual_dev_target
    # do it only missing & possible
    if dev_missing > 0 and dev_slice.shape[0] > 5 and train_slice.shape[0] > 5:
        inx: int = -1
        delta_dev: int = 0
        delta_train: int = 0
        limit_reached: bool = False
        while (delta_train - delta_dev) < dev_missing and not limit_reached:
            inx += 1
            try:
                # start from lowest to higher
                delta_train += int(train_slice["recorded_count"].iat[inx])
                # start from highest to lower
                delta_dev += int(dev_slice["recorded_count"].iat[-inx])
            except ValueError as e:
                limit_reached = True

        delta_dev_df: pd.DataFrame = dev_slice[-inx - 1 :]
        delta_train_df: pd.DataFrame = train_slice[: inx + 1]
        dev_slice: pd.DataFrame = pd.concat([dev_slice[: -inx - 1], delta_train_df])
        train_slice: pd.DataFrame = train_slice[inx + 1 :]

    # Here we have CUT OUT train, now we maybe add the swapped parts
    if delta_dev_df.shape[0] > 0:
        train_slice: pd.DataFrame = pd.concat([delta_dev_df.loc[:], train_slice])

    #
    # STEP-3 : Now, build the actual split DF's
    #

    # Do it again for DEV
    # get a list of unique voice enum in dev set
    dev_voices: "list[str]" = dev_slice["v_enum"].unique().tolist()
    # total_dev_voices: int = len(dev_voices)
    # select all validated records for that list
    dev_df: pd.DataFrame = validated_df[validated_df["v_enum"].isin(dev_voices)]
    # total_dev_sentences: int = len(dev_df["s_enum"].unique().tolist())
    # dev_recs: int = dev_df.shape[0]

    # Rest will be in TRAIN
    train_voices: "list[str]" = train_slice["v_enum"].unique().tolist()
    # total_train_voices: int = len(train_voices)
    train_df: pd.DataFrame = validated_df[validated_df["v_enum"].isin(train_voices)]
    # total_train_sentences: int = len(train_df["s_enum"].unique().tolist())
    # train_recs: int = train_df.shape[0]

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
    print("=== vx algorithm to be used with external test data ===")

    g.start_time = datetime.now()

    # Copy source experiment tree to destination experiment
    src_exppath: str = os.path.join(
        conf.SM_DATA_DIR, "experiments", aspecs.src_algo_dir
    )
    dst_exppath: str = os.path.join(
        conf.SM_DATA_DIR, "experiments", aspecs.dst_algo_dir
    )

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

        if lc not in LANGUAGES_ALLOWED:
            # We need it to be among supported languages
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
    print(
        f"Skipping Existing: {g.skipped_exists} & Not Supported: {g.skipped_nosupport}"
    )

    chunk_size: int = g.src_cnt // PROC_COUNT + 0 if g.src_cnt % PROC_COUNT == 0 else 1

    with mp.Pool(PROC_COUNT) as pool:
        with tqdm(total=g.src_cnt) as pbar:
            for result in pool.imap_unordered(
                algorithm_vx, final_list, chunksize=chunk_size
            ):
                pool_callback(result)

    # remove temp directory structure
    # _ = [shutil.rmtree(d) for d in glob.glob(os.path.join(HERE, ".temp", "*"), recursive=False)]

    final_report(g)


if __name__ == "__main__":
    main()
