#!/usr/bin/env python3
"""
cv-tbox Diversity Check / Split Maker
Alternative splitting algorithm which uses all recordings and regards diversity
"""
###########################################################################
# proposal-v1.py
#
# Alternative proposal for current CorporaCreator implementation v2
#
# It uses validated.tsv to re-create train, dev, test splits
# using unique voice recording frequencies.
#
# A voice only lives in one split.
#
# It takes lowest recorded voices first and put into TEST split
# until predefined percentage (10%) is met OR at max 50% of total voices.
#
# Next it fills the DEV split
# until predefined percentage (10%) is met OR at max 25% of total voices.
#
# The rest will go to TRAIN split
#
# It will use the whole validated recordings.
# The train set will NOT be diverse enough, TEST split will be most diverse.
#
# The script works on multiple CV versions and locales.
#
# The data is grouped as:
# experiments - Common Voice versions - locales - splits
#
# Use:
# python proposal-v1.py
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
import multiprocessing as mp
from multiprocessing.pool import AsyncResult
from datetime import datetime

# External Dependencies
import pandas as pd
import psutil
from tqdm import tqdm

# Module
from lib import Globals, AlgorithmSpecs, AlgorithmResults
from lib import calc_sample_size, df_read, df_write, final_report
import conf

#
# CONST
#
SAMPLE_SIZE_THRESHOLD: int = 150000

#
# Globals
#
HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

PROC_COUNT: int = psutil.cpu_count(logical=True)  # Full usage

g = Globals()
aspecs = AlgorithmSpecs(
    src_algo_dir="s1",
    dst_algo_dir="v1",
    train_percentage=80.0,
    dev_percentage=10.0,
    test_percentage=10.0,
    max_test_user=0.50,
    max_dev_user=0.25,
)


#
# Handle one split creation, this is where calculations happen
#
def algorithm_v1(val_path: str) -> AlgorithmResults:
    """Processes validated.tsv and create new train, dev, test splits"""

    results: AlgorithmResults = AlgorithmResults()

    src_corpus_dir: str = os.path.split(val_path)[0]
    lc: str = os.path.split(src_corpus_dir)[1]
    ver: str = os.path.split(os.path.split(src_corpus_dir)[0])[1]
    dst_path: str = os.path.join(HERE, "experiments", aspecs.dst_algo_dir, ver, lc)
    os.makedirs(dst_path, exist_ok=True)
    results.lc = lc
    results.ver = ver
    # results.ver = ver.replace("cv-corpus-", "")
    if conf.VERBOSE:
        print(f"Executing {ver} - {lc}", flush=True)

    validated_df: pd.DataFrame = df_read(val_path)

    # add lowercase sentence column
    validated_df["sentence_lower"] = validated_df["sentence"].str.lower()
    # add an enumaration column for client_id's
    validated_df["v_enum"], v_unique = pd.factorize(validated_df["client_id"])
    # add an enumaration column for client_id's
    validated_df["s_enum"], s_unique = pd.factorize(validated_df["sentence_lower"])

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
    total_sentences: int = validated_df["s_enum"].max()
    total_voices: int = voices_df.shape[0]

    test_voice_max: int = int(aspecs.max_test_user * total_voices)
    dev_voice_max: int = int(aspecs.max_dev_user * total_voices)

    # Adaptive part - if population size >= 150.000, then use sample size calculation, else use percentages given

    #
    # LARGE
    #
    if total_validated >= SAMPLE_SIZE_THRESHOLD:
        # use sample size calculation
        sample_size: int = int(calc_sample_size(total_validated))
        test_target: int = sample_size
        dev_target: int = sample_size
        train_target: int = total_validated - dev_target - test_target
        results.large = 1

        # if conf.VERBOSE:
        #     print(
        #         f"!!! LARGE DATASET! - Sample sizes for TEST and DEV are recalculated as: {sample_size}"
        #     )
        #     print(
        #         f">>> Processing - {total_validated} validated records with {total_sentences} lower-case unique sentences from {total_voices} voices. Targeting:"
        #     )
        #     print(
        #         f">>> TEST : {dec3(100 * test_target/total_validated)}% => {test_target} recs OR max {test_voice_max} voices"
        #     )
        #     print(
        #         f">>> DEV  : {dec3(100 * dev_target/total_validated)}% => {dev_target} recs OR max {dev_voice_max} voices"
        #     )
        #     print(
        #         f">>> TRAIN: {dec3(100 * train_target/total_validated)}% => {train_target} recs OR remaining from TEST & DEV"
        #     )
        #     # print()
    #
    # MEDIUM
    #
    else:
        # use given percentages
        test_target: int = int(aspecs.test_percentage / 100 * total_validated)
        dev_target: int = int(aspecs.dev_percentage / 100 * total_validated)
        train_target: int = total_validated - dev_target - test_target
        # g.medium_dataset_cnt += 1
        # is_medium = True
        results.medium = 1

        # if conf.VERBOSE:
        #     print(
        #         f">>> Processing - {total_validated} validated records with {total_sentences} lower-case unique sentences from {total_voices} voices. Targeting:"
        #     )
        #     print(
        #         f">>> TEST : {aspecs.test_percentage}% => {test_target} recs OR max {test_voice_max} voices"
        #     )
        #     print(
        #         f">>> DEV  : {aspecs.dev_percentage}% => {dev_target} recs OR max {dev_voice_max} voices"
        #     )
        #     print(
        #         f">>> TRAIN: {aspecs.train_percentage}% => {train_target} recs OR remaining from TEST & DEV"
        #     )
        #     # print()

    #
    # TINY
    #
    if total_validated < 100 or total_voices < 10:
        # g.tiny_dataset_cnt += 1
        # if is_large:
        #     g.large_dataset_cnt -= 1
        # if is_medium:
        #     g.medium_dataset_cnt -= 1
        # if conf.VERBOSE:
        #     print("!!! TOO LOW ON RESOURCES, SPLITTING RANDOMLY !!!")
        # Remove extra columns
        test_df: pd.DataFrame = validated_df[:test_target]
        dev_df: pd.DataFrame = validated_df[test_target + 1 : test_target + dev_target]
        train_df: pd.DataFrame = validated_df[test_target + dev_target :]
        # Writeout results
        df_write(test_df, os.path.join(dst_path, "test.tsv"))
        df_write(dev_df, os.path.join(dst_path, "dev.tsv"))
        df_write(train_df, os.path.join(dst_path, "train.tsv"))
        results.tiny = 1
        results.processed = 1
        return results

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
        # if conf.VERBOSE:
        #     print("TEST-Re-sliced because max voices exceeded")
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
        # if conf.VERBOSE:
        #     print("DEV-Re-sliced because max voices exceeded")
        # Get based on number of voices
        dev_slice: pd.DataFrame = voices_df[0:dev_voice_max]
    actual_dev_target: int = int(dev_slice["cumulative_recordings"].iat[-1])

    # TRAIN
    # Get the rest
    train_slice: pd.DataFrame = voices_df[
        voices_df["cumulative_recordings"].astype(int) > actual_dev_target
    ].reset_index(drop=True)

    # if conf.VERBOSE:
    #     print(f'VOICES: TEST={test_slice.shape[0]}/{test_voice_max}   DEV={dev_slice.shape[0]}/{dev_voice_max}   TRAIN={train_slice.shape[0]} TOTAL={train_slice.shape[0] + dev_slice.shape[0] + test_slice.shape[0]}/{total_voices}')
    #     print(f'ACTUAL: TEST={actual_test_target}/{test_target} DEV={actual_dev_target - actual_test_target}/{dev_target}')

    #
    # STEP-2 : Now swap TEST's high end voices & DEV's high voices end with low end of TRAIN in order to fulfill the target split size.
    #
    # print('SLICES:', test_slice.shape, dev_slice.shape, train_slice.shape)

    delta_test_df: pd.DataFrame = pd.DataFrame(columns=voices_df.columns)
    delta_dev_df: pd.DataFrame = pd.DataFrame(columns=voices_df.columns)
    delta_train_df: pd.DataFrame = pd.DataFrame(columns=voices_df.columns)

    # Handle TEST-TRAIN
    test_missing: int = test_target - actual_test_target  # calc how much missing
    # print('Missing recs in TEST=', test_missing)
    # do it only missing & possible
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
        # here we know
        # if conf.VERBOSE:
        #     print(
        #         f"SWAP TEST-TRAIN {inx+1} VOICES, FOR {delta_train - delta_test} RECORDINGS TO FILL {test_missing} MISSING RECS IN TEST SPLIT"
        #     )
        # print('OLD TRAIN:\n', train_slice.head(inx+2))
        # print('OLD TEST:\n', test_slice.tail(inx+2))
        # Get the tail to move to train (will happen later)
        delta_test_df: pd.DataFrame = test_slice[-inx - 1 :]
        # Get the head to move to test
        delta_train_df: pd.DataFrame = train_slice[: inx + 1]
        # To head of test, append head of train
        test_slice: pd.DataFrame = pd.concat([test_slice[: -inx - 1], delta_train_df])
        # Make a smaller train by removing the moved head
        train_slice: pd.DataFrame = train_slice[inx + 1 :]
        # print('DTEST:\n', delta_test_df.head(inx+2))
        # print('DTRAIN:\n', delta_train_df.head(inx+2))
        # print('NEW TRAIN:\n', train_slice.head(inx+2))
        # print('NEW TEST:\n', test_slice.tail(inx+2))
        # print('DELTAS:', delta_test_df.shape, delta_train_df.shape)
        # print('SLICES:', test_slice.shape, dev_slice.shape, train_slice.shape)

    # Handle DEV-TRAIN
    # calc how much missing
    dev_missing: int = dev_target - (actual_dev_target - actual_test_target)
    # print('Missing recs in DEV=', dev_missing)
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
        # here we know
        # if conf.VERBOSE:
        #     print(
        #         f"SWAP DEV-TRAIN {inx+1} VOICES, FOR {delta_train - delta_dev} RECORDINGS TO FILL {dev_missing} MISSING RECS IN DEV SPLIT"
        #     )
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
    if delta_test_df.shape[0] > 0:
        train_slice: pd.DataFrame = pd.concat([delta_test_df.loc[:], train_slice])

    # print('SLICES:', test_slice.shape, dev_slice.shape, train_slice.shape)
    
    #
    # STEP-3 : Now, build the actual split DF's
    #

    # get a list of unique voice enum in test set
    test_voices: "list[str]" = test_slice["v_enum"].unique().tolist()
    # select all validated records for that list
    test_df: pd.DataFrame = validated_df[validated_df["v_enum"].isin(test_voices)]
    # test_recs: int = test_df.shape[0]
    # # if conf.VERBOSE:
    # #     print(f'--> TEST  split now has {test_recs} records with {total_test_sentences} distinct sentences from {total_test_voices} distinct voices (from {total_voices})')

    # Do it again for DEV
    # get a list of unique voice enum in dev set
    dev_voices: "list[str]" = dev_slice["v_enum"].unique().tolist()
    # get a list of unique voice enum in dev set
    total_dev_voices: int = len(dev_voices)
    # select all validated records for that list
    dev_df: pd.DataFrame = validated_df[validated_df["v_enum"].isin(dev_voices)]
    # total_dev_sentences: int = len(
    #     dev_df["s_enum"].unique().tolist()
    # )  # get a list of unique voice enum in test set
    # dev_recs: int = dev_df.shape[0]
    # # if conf.VERBOSE:
    # #     print(f'--> DEV   split now has {dev_recs} records with {total_dev_sentences} distinct sentences from {total_dev_voices} distinct voices (from {total_voices})')

    # Rest will be in TRAIN
    train_voices: "list[str]" = train_slice["v_enum"].unique().tolist()
    # get a list of unique voice enum in test set
    total_train_voices: int = len(train_voices)
    # get remaining directly from validated
    train_df: pd.DataFrame = validated_df[validated_df["v_enum"].isin(train_voices)]
    # total_train_sentences: int = len(
    #     train_df["s_enum"].unique().tolist()
    # )  # get a list of unique voice enum in test set
    # train_recs: int = train_df.shape[0]
    # if conf.VERBOSE:
    #     # print(f'--> TRAIN split now has {train_recs} records with {total_train_sentences} distinct sentences from {total_train_voices} distinct voices (from {total_voices})')
    #     tot_v: int = total_train_voices + total_dev_voices + total_test_voices
    #     tot_r: int = train_recs + dev_recs + test_recs
    #     print(
    #         f'--> RESULT VOICES  (TRAIN + DEV + TEST) = {total_train_voices} ({dec2(100 * total_train_voices/tot_v)}%) + '
    #         + f'{total_dev_voices} ({dec2(100 * total_dev_voices/tot_v)}%) + '
    #         + f'{total_test_voices} ({dec3(100 * total_test_voices/tot_v)}%) = '
    #         + f"{tot_v} (expecting {total_voices})"
    #     )
    #     print(
    #         f'--> RESULT RECORDS (TRAIN + DEV + TEST) = {train_recs} ({dec2(100 * train_recs/tot_r)}%) + '
    #         + f'{dev_recs} ({dec2(100 * dev_recs/tot_r)}%) + '
    #         + f'{test_recs} ({dec2(100 * test_recs/tot_r)}%) = '
    #         + f"{train_recs + dev_recs + test_recs} (expecting {total_validated})"
    #     )

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

    def pool_callback(result) -> None:
        """Callback to append results and increment bar"""
        # print(f"Finished {res.lc}")
        pbar.update()
        res: AlgorithmResults
        try:
            res = result.get()
            g.processed_cnt += res.processed
            g.skipped_nodata += res.skipped_nodata
            g.skipped_small += res.skipped_small
            g.tiny_dataset_cnt += res.tiny
            g.medium_dataset_cnt += res.medium
            g.large_dataset_cnt += res.large
        except Exception as e:
            print(f"Failed for {result}:", e)

    #
    # Main
    #

    print("=== New Corpora Creator Algorithm Proposal v1 for Common Voice Datasets ===")

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
    if conf.FORCE_CREATE:
        # if force create, we (re-)create all
        final_list = all_validated
    else:
        for val_path in all_validated:
            src_corpus_dir: str = os.path.split(val_path)[0]
            lc: str = os.path.split(src_corpus_dir)[1]
            ver: str = os.path.split(os.path.split(src_corpus_dir)[0])[1]
            dst_corpus_dir: str = os.path.join(dst_exppath, ver, lc)
            if os.path.isfile(os.path.join(dst_corpus_dir, "train.tsv")):
                g.skipped_exists += 1
            else:
                final_list.append(val_path)

    g.total_cnt = len(all_validated)
    g.src_cnt = len(final_list)

    print(
        f"Re-splitting for {g.src_cnt} out of {g.total_cnt} corpora in {PROC_COUNT} processes."
    )
    print(f"Skipping {g.skipped_exists} as they already exist.")

    pbar = tqdm(total=g.src_cnt, unit=" Dataset")
    with mp.Pool(PROC_COUNT) as pool:
        for val_path in final_list:
            res: AsyncResult = pool.apply_async(
                algorithm_v1,
                args=(val_path,),
            )
            # print(res.get())
            pool_callback(res)
        # callback=pool_callback,
        # error_callback=error_callback,
        # for val_path in final_list:
        #     pool_callback(algorithm_v1(val_path))
    pbar.close()

    final_report(g)


if __name__ == "__main__":
    main()
