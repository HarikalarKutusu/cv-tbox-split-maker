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
import numpy as np
import pandas as pd

import corporacreator

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

#
# Constants - TODO These should be arguments
#

# Directories
SRC_ALGO_DIR: str = 's1'
DST_ALGO_DIR: str = 's99'

# Program parameters
VERBOSE: bool = False
FAIL_ON_NOT_FOUND: bool = True

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
    # destination dir
    os.makedirs(dst_path, exist_ok=True)
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
        shutil.move(os.path.join(temp_path, lc, 'train.tsv'), dst_path)
        shutil.move(os.path.join(temp_path, lc, 'dev.tsv'), dst_path)
        shutil.move(os.path.join(temp_path, lc, 'test.tsv'), dst_path)

        res = True

        # remove .temp
        shutil.rmtree(temp_path)
    
    return res

#
# Main loop for experiments-versions-locales
#

def main() -> None:
    print('=== Original Corpora Creator with -s 99 option for Common Voice Datasets ===')

    # Copy source experiment tree to destination experiment
    experiments_path: str = os.path.join(HERE, 'experiments')
    src_exppath: str = os.path.join(experiments_path, SRC_ALGO_DIR)
    dst_exppath: str = os.path.join(experiments_path, DST_ALGO_DIR)
    # shutil.copytree(
    #     src=src_exppath,
    #     dst=dst_exppath,
    #     dirs_exist_ok=True,
    #     ignore=shutil.ignore_patterns('*.*')
    #     )

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

        if os.path.isfile(os.path.join(dst_corpus_dir, 'train.tsv')):
            # Already there, so skip
            cnt_skipped += 1
        else:
            if not corpora_creator_original(
                    lc=lc,
                    val_path=val_path,
                    dst_path=dst_corpus_dir,
                    duplicate_sentences=99
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
        avg_seconds_new: float = process_seconds/cnt_processed
    print('\n' + '-' * 80)
    print(f'Finished processing of {cnt_total} corpora in {str(process_timedelta)}, avg duration {float("{:.3f}".format(avg_seconds))}')
    print(f'Processed: {cnt}, Skipped: {cnt_skipped}, New: {cnt_processed}')
    if cnt_processed > 0:
        print(f'Avg. time new split creation: {float("{:.3f}".format(avg_seconds_new))}')

main()
