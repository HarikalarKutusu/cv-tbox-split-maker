 #!/usr/bin/env python3

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

import sys, os, shutil, glob, csv
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Literal
import numpy as np
import pandas as pd
import av
# Get rid of warnings
import logging
logging.getLogger('libav').setLevel(logging.ERROR)
# Algo
import corporacreator

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

#
# Constants - TODO These should be arguments
#

DUPLICATE_SENTENCE_COUNT: int = 1

# Directories
SRC_ALGO_DIR: str = 's1'
DST_ALGO_DIR: str = 's1'

SOURCE_DATASET_DIR: str = os.path.normpath("M:\\DATASETS\\CV")
SOURCE_DATASET: str = "cv-corpus-13.0-2023-03-09"
USE_SOURCE_DATASET_DIR: bool = True
DO_CALC_CLIP_DURATIONS: bool = True

# Program parameters
VERBOSE: bool = False               # If true, report all on different lines, else show only generated
FAIL_ON_NOT_FOUND: bool = True      # If true, fail if source is not found, else skip it
FORCE_CREATE: bool = False          # If true, regenerate the splits even if they exist

# DF related (fır clip durations)
DF_COLS: list[str] = ['clip', 'duration']
DF_FN: Literal['$clip_durations.tsv'] = '$clip_durations.tsv'
DF_ERR_COLS: list[str] = ['clip', 'error']
DF_ERR_FN: Literal['$clip_durations_errors.tsv'] = '$clip_durations_errors.tsv'

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
    numerator: float = (z_score ** 2 * fraction_picking * (1 - fraction_picking)) / (
        margin_of_error ** 2
    )
    denominator: float = 1 + (z_score ** 2 * fraction_picking * (1 - fraction_picking)) / (
        margin_of_error ** 2 * population_size
    )
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
        """Creates a :class:`corporacreator.Corpus` for `self.locale`.
        """
        self._post_process_valid_data()

    def _post_process_valid_data(self):
        # Remove duplicate sentences while maintaining maximal user diversity at the frame's start (TODO: Make addition of user_sentence_count cleaner)
        speaker_counts = self.validated["client_id"].value_counts()
        speaker_counts = speaker_counts.to_frame().reset_index()
        speaker_counts.columns = ["client_id", "user_sentence_count"]
        self.validated = self.validated.join(
            speaker_counts.set_index("client_id"), on="client_id"
        )
        self.validated = self.validated.sort_values(["user_sentence_count", "client_id"])
        validated = self.validated.groupby("sentence").head(self.args.duplicate_sentence_count)

        validated = validated.sort_values(["user_sentence_count", "client_id"], ascending=False)
        validated = validated.drop(columns="user_sentence_count")
        self.validated = self.validated.drop(columns="user_sentence_count")


        train = pd.DataFrame(columns=validated.columns)
        dev = pd.DataFrame(columns=validated.columns)
        test = pd.DataFrame(columns=validated.columns)

        train_size = dev_size = test_size = 0

        if (len(validated) > 0):
            # Determine train, dev, and test sizes
            train_size, dev_size, test_size = self._calculate_data_set_sizes(len(validated))
            # Split into train, dev, and test datasets
            continous_client_index, uniques = pd.factorize(validated["client_id"])
            validated["continous_client_index"] = continous_client_index

            for i in range(max(continous_client_index), -1, -1):
                if len(test) + len(validated[validated["continous_client_index"] == i]) <= test_size:
                    test = pd.concat([test, validated[validated["continous_client_index"] == i]], sort=False)
                elif len(dev) + len(validated[validated["continous_client_index"] == i]) <= dev_size:
                    dev = pd.concat([dev, validated[validated["continous_client_index"] == i]], sort=False)
                else:
                    train = pd.concat([train, validated[validated["continous_client_index"] == i]], sort=False)

        self.train = train.drop(columns="continous_client_index", errors="ignore")
        self.dev = dev.drop(columns="continous_client_index", errors="ignore")
        self.test = test[:train_size].drop(columns="continous_client_index", errors="ignore")

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
            path, sep="\t", header=True, index=False, encoding="utf-8", escapechar='\\', quoting=csv.QUOTE_NONE
        )

#
# Handle one split creation, this is where calculations happen
#

def corpora_creator_original(lc: str, val_path: str, dst_path: str, duplicate_sentences: int) -> bool:
    """Processes validated.tsv and create new train, dev, test splits"""

    # Assume result false
    res: bool = False
    # temp dir
    temp_path: str = os.path.join(HERE, '.temp')

    # call corpora creator with only validated (we don't need others)
    df_corpus: pd.DataFrame = df_read(val_path)

    # Must have records in it
    if df_corpus.shape[0] > 0:
        # create temp dir
        os.makedirs(temp_path, exist_ok=True)

        # handle corpus
        args = corporacreator.parse_args(
            ['-d', temp_path, '-f', val_path, '-s', str(duplicate_sentences)]
            )
        corpus: LocalCorpus = LocalCorpus(args, lc, df_corpus)
        corpus.validated = df_corpus
        corpus.create()
        corpus.save(temp_path)

        # move required files to destination
        os.makedirs(dst_path, exist_ok=True)
        shutil.move(os.path.join(temp_path, lc, 'train.tsv'), dst_path)
        shutil.move(os.path.join(temp_path, lc, 'dev.tsv'), dst_path)
        shutil.move(os.path.join(temp_path, lc, 'test.tsv'), dst_path)
        shutil.rmtree(temp_path)

        res = True
    
    return res

#
# Main loop for experiments-versions-locales
#
# Main Loop for Clips
def build_clip_durations_table(srcdir):
    start: datetime = datetime.now()
    # get list
    mp3list: list[str] = glob.glob(os.path.join(srcdir, '*.mp3'))
    mp3list.sort()
    num_files: int = len(mp3list)
    log_step: int = int(num_files / 10)
    if log_step == 0:
      log_step = 1
    # process list
    cnt: int = 0
    skipped: int = 0
    total_dur: float = 0
    perc = 0
    # Start display
    print(f'Creating $clip_durations.tsv table for {num_files} clips into "{srcdir}"')
    print('+' * perc + '.' * (100-perc) + f' {perc}% - {cnt}/{num_files} => {0.00} hours.')
    data: list[Any] = []
    data_err: list[Any] = []
    a: Any = None
    for fn in mp3list:
        cnt += 1
        perc: int = int(100*cnt/num_files + 0.5)
        if os.path.getsize(fn) == 0:
            print(f'ERROR: Zero filesize  - {fn}')
            skipped += 1
            data_err.append([
                os.path.split(fn)[-1],
                'zero_filesize'
            ])
            continue # skip if filesize is 0
        else:
            err: bool = False
            try:
                a = av.open(fn)
            except:
                print(f'ERROR: During opening - {fn}')
                data_err.append([
                    os.path.split(fn)[-1],
                    'could_not_open'
                ])
                skipped += 1
                err: bool = True
        if not err and a:
          file_duration: float = (a.duration)/1000000
          total_dur += file_duration
          if (cnt % log_step == 0):
            print('+' * perc + '.' * (100-perc) + f' {perc}% - {cnt}/{num_files} => {round(total_dur/3600,2)} hours.')
          # add to list
          data.append([
              os.path.split(fn)[-1],
              file_duration
          ])

    # finish
    finish: datetime = datetime.now()
    pduration: timedelta = finish - start
    pduration_sec: int = int(pduration.total_seconds())
    if skipped > 0:
        print(f'Skipped {skipped} files due to errors.')
    print(f'Finished {num_files} files in {pduration} sec, avg= {pduration_sec/num_files}.')
    print(f'Total audio duration {round(total_dur/3600,2)} hours, avg. duration= {total_dur/(num_files-skipped)} sec.')
    # Build dataframe and save
    df: pd.DataFrame = pd.DataFrame(data, columns=DF_COLS).reset_index(drop=True)
    df_write(df, fpath=os.path.join(srcdir, DF_FN))
    if len(data_err) > 0:
      df_err: pd.DataFrame = pd.DataFrame(data_err, columns=DF_ERR_COLS).reset_index(drop=True)
      df_write(df, fpath=os.path.join(srcdir, DF_ERR_FN))


#
# Main loop for experiments-versions-locales
#

def main() -> None:
    print('=== Original Corpora Creator with -s 1 option for Common Voice Datasets (if splits are not provided) ===')

    # Copy source experiment tree to destination experiment
    experiments_path: str = os.path.join(HERE, 'experiments')
    src_exppath: str = os.path.join(experiments_path, SRC_ALGO_DIR)
    dst_exppath: str = os.path.join(experiments_path, DST_ALGO_DIR)

    # Calculate clip durations?
    if DO_CALC_CLIP_DURATIONS:
        print('=== REFRESH CLIP DURATIONS ===')
        # remove existing clip durations
        old_clip_durations: list[str] = glob.glob(os.path.join(experiments_path, '**', '$clip_durations.tsv'), recursive=True)
        print(f'=== Remove existing {len(old_clip_durations)} files...')
        for clip_path in old_clip_durations:
            os.remove(path=clip_path)
        # recalculate clip durations
        clips_dirs: list[str] = glob.glob(os.path.join(SOURCE_DATASET_DIR, SOURCE_DATASET, '**', 'clips'), recursive=True)
        print(f'=== Processing {len(clips_dirs)} locales')
        for clips_dir in clips_dirs:
            build_clip_durations_table(clips_dir)

    # Do we want to copy the .tsv files from original expanded datasets?
    if USE_SOURCE_DATASET_DIR:
        # copy all .tsv files while forming structure
        print('=== COPY .TSV FILES FROM DATASETS ===')
        copyto_corpus_dir: str = os.path.join(src_exppath, SOURCE_DATASET)
        os.makedirs(name=copyto_corpus_dir, exist_ok=True)
        shutil.copytree(
            src=os.path.join(SOURCE_DATASET_DIR, SOURCE_DATASET),
            dst=copyto_corpus_dir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns('*.mp3')
            )

    # !!! from now on we will work on destination !!!

    # src_corpora_paths: "list[str]" = glob.glob(os.path.join(src_exppath, '*'), recursive=False)
    # dst_corpora_paths: "list[str]" = glob.glob(os.path.join(dst_exppath, '*'), recursive=False)

    # Get total for progress display
    all_validated: "list[str]" = glob.glob(os.path.join(src_exppath, '**', 'validated.tsv'), recursive=True)
    print(f'Re-splitting for {len(all_validated)} corpora... Wait for final structure is formed...')
    print()   # extra line is for progress line

    # For each corpus
    cnt: int = 0            # count of corpora checked
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
            print(f'\n=== Processing {cnt}/{cnt_total} => {ver} - {lc}')
        else:
            print('\033[F' + ' ' * 80)
            print(f'\033[FProcessing {cnt}/{cnt_total} => {ver} - {lc}')

        if not FORCE_CREATE and os.path.isfile(os.path.join(dst_corpus_dir, 'train.tsv')):
            # Already there and is not forced to recreate, so skip
            cnt_skipped += 1
        else:
            if not corpora_creator_original( # df might be empty, thus returns false
                    lc=lc,
                    val_path=val_path,
                    dst_path=dst_corpus_dir,
                    duplicate_sentences=DUPLICATE_SENTENCE_COUNT
                    ):
                cnt_skipped += 1
            print()

    finish_time: datetime = datetime.now()
    process_timedelta: timedelta = finish_time - start_time
    process_seconds: float = process_timedelta.total_seconds()
    avg_seconds: float = process_seconds/cnt_total
    cnt_processed: int = cnt-cnt_skipped
    avg_seconds_new: float = -1
    if cnt_processed > 0:
        avg_seconds_new = process_seconds/cnt_processed
    print('\n' + '-' * 80)
    print(f'Finished processing of {cnt_total} corpora in {str(process_timedelta)} secs, avg duration {float("{:.3f}".format(avg_seconds))} secs')
    print(f'Processed: {cnt}, Skipped: {cnt_skipped}, New: {cnt_processed}')
    if cnt_processed > 0:
        print(f'Avg. time new split creation: {float("{:.3f}".format(avg_seconds_new))} secs')

main()
