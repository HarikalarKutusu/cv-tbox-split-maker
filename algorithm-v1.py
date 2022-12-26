 #!/usr/bin/env python3

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

import sys, os, shutil, glob, csv
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

#
# Constants - TODO These should be arguments
#

# Real Constants
SAMPLE_SIZE_THRESHOLD: int = 150000

# Directories
SRC_ALGO_DIR: str = 's1'
DST_ALGO_DIR: str = 'v1'


TRAIN_PERCENTAGE: float = 80.0
DEV_PERCENTAGE: float   = 10.0
TEST_PERCENTAGE: float  = 10.0

MAX_TEST_USER: float = 0.50
MAX_DEV_USER: float  = 0.25

# Program parameters
VERBOSE: bool = False               # If true, report all on different lines, else show only generated
FAIL_ON_NOT_FOUND: bool = True      # If true, fail if source is not found, else skip it
FORCE_CREATE: bool = False          # If true, regenerate the splits even if they exist

#
# DataFrame file read-write 
#

def df_read(fpath: str) -> pd.DataFrame:
    """Read a tsv file into a dataframe"""
    if not os.path.isfile(fpath):
        print(f'FATAL: File {fpath} cannot be located!')
        if FAIL_ON_NOT_FOUND:
            sys.exit(1)
    
    df: pd.DataFrame = pd.read_csv(
        fpath,
        sep="\t",
        parse_dates=False,
        engine="python",
        encoding="utf-8",
        on_bad_lines='skip',
        quotechar='"',
        quoting=csv.QUOTE_NONE,
    )
    return df


def df_write(df: pd.DataFrame, fpath: str) -> None:
    """Write dataframe to a tsv file"""
    df.to_csv(fpath, header=True, index=False, encoding="utf-8", sep='\t', escapechar='\\', quoting=csv.QUOTE_NONE)

#
# Handle one split creation, this is where calculations happen
#

def corpora_creator_v1(val_path: str, dst_path: str):
    """Processes validated.tsv and create new train, dev, test splits"""

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
        numerator: float = (z_score ** 2 * fraction_picking * (1 - fraction_picking)) / (
            margin_of_error ** 2
        )
        denominator: float = 1 + (z_score ** 2 * fraction_picking * (1 - fraction_picking)) / (
            margin_of_error ** 2 * population_size
        )
        return numerator / denominator



    validated_df: pd.DataFrame = df_read(val_path)

    # add lowercase sentence column
    validated_df['sentence_lower'] = validated_df['sentence'].str.lower()
    validated_df['v_enum'], v_unique = pd.factorize(validated_df['client_id']) # add an enumaration column for client_id's
    validated_df['s_enum'], s_unique = pd.factorize(validated_df['sentence_lower']) # add an enumaration column for client_id's

    # Try with unique voices
    voices_df: pd.DataFrame = validated_df.groupby('v_enum').agg( {'path': 'count', 's_enum': 'count'} ).reset_index()    # get list with count agg
    voices_df.rename(columns= {'path': 'recorded_count', 'client_id': 'sentence_count'}, inplace=True )                   # rename agg column
    voices_df.sort_values(by=['recorded_count', 'v_enum'], ascending=True, inplace=True)                                  # sort in ascending recorded count
    voices_df['cumulative_recordings'] = voices_df['recorded_count'].cumsum()                                                        # add a cumulative sum for easy access
    # sentences_df['s_enum'], s_unique = pd.factorize(sentences_df['sentence_lower'])                                     # add an enumaration column for sentences (lower)
    voices_df.reset_index()

    # CALCULATE split sizes as record counts
    total_validated: int    = validated_df.shape[0]
    total_sentences: int    = validated_df['s_enum'].max()
    total_voices: int       = voices_df.shape[0]

    test_voice_max: int  = int(MAX_TEST_USER * total_voices)
    dev_voice_max: int   = int(MAX_DEV_USER  * total_voices)

    # Adaptive part - if population size >= 150.000, then use sample size calculation, else use percentages given

    if total_validated >= SAMPLE_SIZE_THRESHOLD:
        # use sample size calculation
        sample_size: int = int(calc_sample_size(total_validated))
        test_target: int  = sample_size
        dev_target: int   = sample_size
        train_target: int = total_validated - dev_target - test_target
        if VERBOSE:
            print(f'!!! LARGE DATASET! - Sample sizes for TEST and DEV are recalculated as: {sample_size}')
            print(f'>>> Processing - {total_validated} validated records with {total_sentences} lower-case unique sentences from {total_voices} voices. Targeting:')
            print(f'>>> TEST : {float("{:.2f}".format(100 * test_target/total_validated))}% => {test_target} recs OR max {test_voice_max} voices')
            print(f'>>> DEV  : {float("{:.2f}".format(100 * dev_target/total_validated))}% => {dev_target} recs OR max {dev_voice_max} voices')
            print(f'>>> TRAIN: {float("{:.2f}".format(100 * train_target/total_validated))}% => {train_target} recs OR remaining from TEST & DEV')
            # print()
    else:
        # use given percentages
        test_target: int  = int(TEST_PERCENTAGE / 100 * total_validated)
        dev_target: int   = int(DEV_PERCENTAGE / 100 * total_validated)
        train_target: int = total_validated - dev_target - test_target
        if VERBOSE:
            print(f'>>> Processing - {total_validated} validated records with {total_sentences} lower-case unique sentences from {total_voices} voices. Targeting:')
            print(f'>>> TEST : {TEST_PERCENTAGE}% => {test_target} recs OR max {test_voice_max} voices')
            print(f'>>> DEV  : {DEV_PERCENTAGE}% => {dev_target} recs OR max {dev_voice_max} voices')
            print(f'>>> TRAIN: {TRAIN_PERCENTAGE}% => {train_target} recs OR remaining from TEST & DEV')
            # print()

    if total_validated < 100 or total_voices < 10:
        print('!!! TOO LOW ON RESOURCES, SPLITTING RANDOMLY !!!')
        # Remove extra columns
        test_df: pd.DataFrame   = validated_df[ : test_target]
        dev_df: pd.DataFrame    = validated_df[test_target+1 : test_target + dev_target]
        train_df: pd.DataFrame  = validated_df[test_target + dev_target : ]
        # Writeout results
        df_write(test_df, os.path.join(dst_path, 'test.tsv'))
        df_write(dev_df, os.path.join(dst_path, 'dev.tsv'))
        df_write(train_df, os.path.join(dst_path, 'train.tsv'))
        return



    #
    # STEP-1 : First run to predict slices for splits
    #
    
    # TEST

    # Test use cumsum to get them quickly
    test_slice: pd.DataFrame = voices_df[ voices_df['cumulative_recordings'].astype(int) <= test_target ].reset_index(drop=True) # use cumulative column to get list of user recordings to match the amount
    # If sliced records are more than test_voice_max we need to re-slice
    if test_slice.shape[0] > test_voice_max:
        if VERBOSE:
            print('TEST-Re-sliced because max voices exceeded')
        test_slice: pd.DataFrame = voices_df[0:test_voice_max]                                                      # This time get the first N voices
    actual_test_target: int = int(test_slice['cumulative_recordings'].iat[-1])

    # DEV
    voices_df: pd.DataFrame = voices_df[ test_slice.shape[0] : ]
    dev_slice: pd.DataFrame = voices_df[ voices_df['cumulative_recordings'].astype(int) <= actual_test_target + dev_target ].reset_index(drop=True)  # use cumulative column to get list of user recordings to match the amount
    if dev_slice.shape[0] > dev_voice_max:
        if VERBOSE:
            print('DEV-Re-sliced because max voices exceeded')
        dev_slice: pd.DataFrame = voices_df[0 : dev_voice_max]                                                     # Get based on number of voices
    actual_dev_target: int = int(dev_slice['cumulative_recordings'].iat[-1])

    # TRAIN

    train_slice: pd.DataFrame = voices_df[ voices_df['cumulative_recordings'].astype(int) > actual_dev_target ].reset_index(drop=True)     # Get the rest

    # if VERBOSE:
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
    test_missing: int = test_target - actual_test_target                # calc how much missing
    # print('Missing recs in TEST=', test_missing)
    if test_missing > 0 and test_slice.shape[0] > 5 and train_slice.shape[0] > 5: # do it only missing & possible
        inx: int = -1
        delta_test: int = 0
        delta_train: int = 0
        while (delta_train - delta_test) < test_missing:
            inx += 1
            delta_train += int(train_slice['recorded_count'].iat[inx])        # start from lowest to higher
            delta_test += int(test_slice['recorded_count'].iat[-inx])         # start from highest to lower
            # print('...step...', inx, delta_train, delta_test)
        # here we know
        if VERBOSE:
            print(f'SWAP TEST-TRAIN {inx+1} VOICES, FOR {delta_train - delta_test} RECORDINGS TO FILL {test_missing} MISSING RECS IN TEST SPLIT')
        # print('OLD TRAIN:\n', train_slice.head(inx+2))
        # print('OLD TEST:\n', test_slice.tail(inx+2))
        delta_test_df: pd.DataFrame = test_slice[-inx-1 : ]                                   # Get the tail to move to train (will happen later)
        delta_train_df: pd.DataFrame = train_slice[ : inx+1]                                     # Get the head to move to test
        test_slice: pd.DataFrame = pd.concat([test_slice[ :-inx-1], delta_train_df])        # To head of test, append head of train
        train_slice: pd.DataFrame = train_slice[inx+1 : ]                                         # Make a smaller train by removing the moved head
        # print('DTEST:\n', delta_test_df.head(inx+2))
        # print('DTRAIN:\n', delta_train_df.head(inx+2))
        # print('NEW TRAIN:\n', train_slice.head(inx+2))
        # print('NEW TEST:\n', test_slice.tail(inx+2))
        # print('DELTAS:', delta_test_df.shape, delta_train_df.shape)
        # print('SLICES:', test_slice.shape, dev_slice.shape, train_slice.shape)

    # Handle DEV-TRAIN
    dev_missing: int = dev_target - (actual_dev_target - actual_test_target)             # calc how much missing
    # print('Missing recs in DEV=', dev_missing)
    if dev_missing > 0 and dev_slice.shape[0] > 5 and train_slice.shape[0] > 5: # do it only missing & possible
        inx: int = -1
        delta_dev: int = 0
        delta_train: int = 0
        while (delta_train - delta_dev) < dev_missing:
            inx += 1
            delta_train += int(train_slice['recorded_count'].iat[inx])        # start from lowest to higher
            delta_dev += int(dev_slice['recorded_count'].iat[-inx])         # start from highest to lower
            # print('...step...', inx, delta_train, delta_dev)
        # here we know
        if VERBOSE:
            print(f'SWAP DEV-TRAIN {inx+1} VOICES, FOR {delta_train - delta_dev} RECORDINGS TO FILL {dev_missing} MISSING RECS IN DEV SPLIT')
        # print('OLD TRAIN:\n', train_slice.head(inx+2))
        # print('OLD DEV:\n', dev_slice.tail(inx+2))
        delta_dev_df: pd.DataFrame = dev_slice[-inx-1 : ]
        delta_train_df: pd.DataFrame = train_slice[ : inx+1]
        dev_slice: pd.DataFrame = pd.concat([dev_slice[ :-inx-1], delta_train_df])
        train_slice: pd.DataFrame = train_slice[inx+1 : ]
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

    test_voices: "list[str]" = test_slice['v_enum'].unique().tolist()                                  # get a list of unique voice enum in test set
    total_test_voices: int = len(test_voices)                                                       # get a list of unique voice enum in test set
    test_df: pd.DataFrame = validated_df[ validated_df['v_enum'].isin(test_voices) ]                # select all validated records for that list
    total_test_sentences: int = len(test_df['s_enum'].unique().tolist())                            # get a list of unique voice enum in test set
    test_recs: int = test_df.shape[0]
    # if VERBOSE:
    #     print(f'--> TEST  split now has {test_recs} records with {total_test_sentences} distinct sentences from {total_test_voices} distinct voices (from {total_voices})')

    # Do it again for DEV
    dev_voices: "list[str]" = dev_slice['v_enum'].unique().tolist()                                                        # get a list of unique voice enum in dev set
    total_dev_voices: int = len(dev_voices)                                                                             # get a list of unique voice enum in dev set
    dev_df: pd.DataFrame = validated_df[ validated_df['v_enum'].isin(dev_voices) ]                                      # select all validated records for that list
    total_dev_sentences: int = len(dev_df['s_enum'].unique().tolist())                                                  # get a list of unique voice enum in test set
    dev_recs: int = dev_df.shape[0]
    # if VERBOSE:
    #     print(f'--> DEV   split now has {dev_recs} records with {total_dev_sentences} distinct sentences from {total_dev_voices} distinct voices (from {total_voices})')

    # Rest will be in TRAIN
    train_voices: "list[str]" = train_slice['v_enum'].unique().tolist()
    total_train_voices: int = len(train_voices)                                                                         # get a list of unique voice enum in test set
    train_df: pd.DataFrame = validated_df[ validated_df['v_enum'].isin(train_voices) ]                                  # get remaining directly from validated
    total_train_sentences: int = len(train_df['s_enum'].unique().tolist())                                              # get a list of unique voice enum in test set
    train_recs: int = train_df.shape[0]
    if VERBOSE:
        # print(f'--> TRAIN split now has {train_recs} records with {total_train_sentences} distinct sentences from {total_train_voices} distinct voices (from {total_voices})')
        tot_v: int = total_train_voices + total_dev_voices + total_test_voices
        tot_r: int = train_recs + dev_recs + test_recs
        print(f'--> RESULT VOICES  (TRAIN + DEV + TEST) = {total_train_voices} ({float("{:.2f}".format(100 * total_train_voices/tot_v))}%) + '
            + f'{total_dev_voices} ({float("{:.2f}".format(100 * total_dev_voices/tot_v))}%) + '
            + f'{total_test_voices} ({float("{:.4f}".format(100 * total_test_voices/tot_v))}%) = '
            + f'{tot_v} (expecting {total_voices})')
        print(f'--> RESULT RECORDS (TRAIN + DEV + TEST) = {train_recs} ({float("{:.2f}".format(100 * train_recs/tot_r))}%) + '
            + f'{dev_recs} ({float("{:.2f}".format(100 * dev_recs/tot_r))}%) + '
            + f'{test_recs} ({float("{:.2f}".format(100 * test_recs/tot_r))}%) = '
            + f'{train_recs + dev_recs + test_recs} (expecting {total_validated})')

    # Remove extra columns
    test_df: pd.DataFrame = test_df.drop(columns=['sentence_lower', 'v_enum', 's_enum'], errors="ignore")     # drop temp columns
    dev_df: pd.DataFrame = dev_df.drop(columns=['sentence_lower', 'v_enum', 's_enum'], errors="ignore")                 # drop temp columns
    train_df: pd.DataFrame = train_df.drop(columns=['sentence_lower', 'v_enum', 's_enum'], errors="ignore")             # drop temp columns

    # Writeout results
    df_write(test_df, os.path.join(dst_path, 'test.tsv'))
    df_write(dev_df, os.path.join(dst_path, 'dev.tsv'))
    df_write(train_df, os.path.join(dst_path, 'train.tsv'))

    # done
    # sys.exit()

#
# Main loop for experiments-versions-locales
#

def main() -> None:
    print('=== New Corpora Creator Algorithm Proposal v1 for Common Voice Datasets ===')

    # Copy source experiment tree to destination experiment
    experiments_path: str = os.path.join(HERE, 'experiments')
    src_exppath: str = os.path.join(experiments_path, SRC_ALGO_DIR)
    dst_exppath: str = os.path.join(experiments_path, DST_ALGO_DIR)
    # shutil.copytree(src=src_exppath, dst=dst_exppath, dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.tsv'))

    # !!! from now on we will work on destination !!!

    # src_corpora_paths: "list[str]" = glob.glob(os.path.join(src_exppath, '*'), recursive=False)
    # dst_corpora_paths: "list[str]" = glob.glob(os.path.join(dst_exppath, '*'), recursive=False)

    # Get total for progress display
    all_validated: "list[str]" = glob.glob(os.path.join(src_exppath, '**', 'validated.tsv'), recursive=True)
    print(f'Re-splitting for {len(all_validated)} corpora... Wait for final structure is formed...')
    print()   # extra line is for progress line

    # For each corpus
    cnt: int = 0 # counter of corpora done
    cnt_skipped: int = 0    # count of corpora skipped
    start_time: datetime = datetime.now()

    cnt_total: int = len(all_validated)

    for val_path in all_validated:
        src_corpus_dir: str = os.path.split(val_path)[0]
        lc: str = os.path.split(src_corpus_dir)[1]
        ver: str = os.path.split(os.path.split(src_corpus_dir)[0])[1]
        dst_corpus_dir: str = os.path.join(dst_exppath, ver, lc)

        cnt += 1
        if VERBOSE:
            print(f'\n=== Processing {cnt}/{cnt_total} => {ver} - {lc}\n')
        else:
            print('\033[F' + ' ' * 80)
            print(f'\033[FProcessing {cnt}/{cnt_total} => {ver} - {lc}')
        
        if not FORCE_CREATE and os.path.isfile(os.path.join(dst_corpus_dir, 'train.tsv')):
            # Already there and is not forced to recreate, so skip
            cnt_skipped += 1
        else:
            os.makedirs(dst_corpus_dir, exist_ok=True)
            corpora_creator_v1(val_path=val_path, dst_path=dst_corpus_dir)
            print()


    finish_time: datetime = datetime.now()
    process_timedelta: timedelta = finish_time - start_time
    process_seconds: float = process_timedelta.total_seconds()
    avg_seconds: float = process_seconds/cnt_total
    cnt_processed: int = cnt-cnt_skipped
    avg_seconds_new: float = -1
    if cnt_processed > 0:
        avg_seconds_new: float = process_seconds/cnt_processed
    print('\n' + '-' * 80)
    print(f'Finished processing of {cnt_total} corpora in {str(process_timedelta)}, avg duration {float("{:.3f}".format(avg_seconds))}')
    print(f'Processed: {cnt}, Skipped: {cnt_skipped}, New: {cnt_processed}')
    if cnt_processed > 0:
        print(f'Avg. time new split creation: {float("{:.3f}".format(avg_seconds_new))}')

main()
