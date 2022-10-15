 #!/usr/bin/env python3

###########################################################################
# tbox_diversity_table.py
#
# Compiles a table of dataset split files with basic info for further analysis
# You need to prepare a line of experiment, containing .tsv files
# These .tsv files can come from original datasets (default splits), from
# CorporaCreator runs with different -s parameters, or just other splits 
# that you prepare with any method. 
# Put different splitting algorithms into different "experiments".
# 
# The data is grouped as:
# experiments - Common Voice versions - locales - splits
# 
# Use:
# cvc_convert [required_parameters] [optional_parameters]
# 
# This script is part of Common Voice ToolBox Package
# 
# [github]
# [copyright]
###########################################################################

from genericpath import isfile
import sys, os, glob, csv
from pathlib import Path
# from datetime import datetime
import numpy as np
import pandas as pd

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

# Constants
SPLITS_IDS: "list[str]" = ['validated', 'train', 'dev', 'test']
RESULT_COLS: "list[str]" = [
    'experiment', 'version', 'locale', 'split',     # All together, these point to a dataset split
    # All other measures below are per split
    'clips',                                        # Number of clips
    'unique_voices',                                # Number of unique voices
    'unique_sentences',                             # Number of unique sentences
    'unique_sentences_lower',                       # Number of unique sentences when using lower()
    'duplicate_sentences',                          # Number of non-unique sentences (multiple recordings per sentence)
    'duplicate_sentences_lower',                    # Number of non-unique sentences when using lower()
    'duplicate_ratio',                              # ratio of 
    'duplicate_ratio_lower',                        # ratio of

    # The following are counts of recordings per gender (not individuals)
    'genders_nodata', 'genders_male', 'genders_female', 'genders_other',
    # The following are counts of recordings per age group (not individuals)
    'ages_nodata', 'ages_teens', 'ages_twenties', 'ages_thirties', 'ages_fourties', 'ages_fifties', 'ages_sixties', 'ages_seventies', 'ages_eighties', 'ages_nineties',
    # Some extra measures
    'genders_f_m_ratio',                            # Female/male ratio N=recordings
    # 'genders_unique_nodata', 'genders_unique_males', 'genders_unique_females', 'genders_unique_other'

]

NODATA: str             = 'nodata'    # .isna cases replaced with this
CV_GENDERS: "list[str]" = [NODATA, 'male', 'female', 'other']
CV_AGES: "list[str]"    = [NODATA, 'teens', 'twenties', 'thirties', 'fourties', 'fifties', 'sixties', 'seventies', 'eighties', 'nineties']

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
# Handle one split, this is where calculations happen
#

def handle_split(exp: str, ver: str, lc: str, split: str, fpath: str):
    """Processes a single split and returns calculated values"""

    def _get_row_total(pt: pd.DataFrame, lbl: str) -> int:
        return int(pd.to_numeric(pt.loc[lbl, 'TOTAL'])) if lbl in list(pt.index.values) else int(0)

    def _get_col_total(pt: pd.DataFrame, lbl: str) -> int:
        return int(pd.to_numeric(pt.loc['TOTAL', lbl])) if lbl in list(pt.columns.values) else int(0)

    df: pd.DataFrame = df_read(fpath)
    # Do nothing, if there is no data
    if df.shape[0] == 0:
        results = {
            'experiment':               exp,
            'version':                  ver,
            'locale':                   lc,
            # 'split':                    split,
        }
        return results

    # The default column structure of CV dataset splits is as follows
    # client_id, path, sentence, up_votes, down_votes, age, gender, accents, locale, segment

    # we have as input:
    # 'experiment', 'version', 'locale', 'split'

    # now, do calculate some statistics. We need:
    # 'clips', 'unique_voices', 'unique_sentences', 'duplicate_sentences',
    # 'genders_nodata', 'genders_male', 'genders_female', 'genders_other',
    # 'ages_nodata', 'ages_teens', 'ages_twenties', 'ages_thirties', 'ages_fourties', 'ages_fifties', 'ages_sixties', 'ages_seventies', 'ages_eighties', 'ages_nineties'

    # Replace NA with NODATA
    df.fillna(value=NODATA, inplace=True)

    # add lowercase sentence column
    df['sentence_lower'] = df['sentence'].str.lower()

    # basic measures
    clips_cnt: int                      = df.shape[0]
    unique_voices: int                  = df['client_id'].unique().shape[0]
    unique_sentences: int               = df['sentence'].unique().shape[0]
    unique_sentences_lower: int         = df['sentence_lower'].unique().shape[0]
    duplicate_sentence_cnt: int         = clips_cnt - unique_sentences
    duplicate_sentence_cnt_lower: int   = clips_cnt - unique_sentences_lower

    # get a pt for all demographics
    _pt: pd.DataFrame = pd.pivot_table(df, values='path', index=['age'], columns=['gender'], aggfunc='count', fill_value=0, dropna=False, margins=True, margins_name='TOTAL' )

    _males: int                         = _get_col_total(_pt, 'male')
    _females: int                       = _get_col_total(_pt, 'female')

    results = {
        'experiment':               exp,
        'version':                  ver,
        'locale':                   lc,
        'split':                    split,

        'clips':                    clips_cnt,
        'unique_voices':            unique_voices,
        'unique_sentences':         unique_sentences,
        'unique_sentences_lower':   unique_sentences_lower,
        'duplicate_sentences':      duplicate_sentence_cnt,
        'duplicate_sentences_lower': duplicate_sentence_cnt_lower,
        'duplicate_ratio':          float("{:.4f}".format(duplicate_sentence_cnt / unique_sentences)),
        'duplicate_ratio_lower':    float("{:.4f}".format(duplicate_sentence_cnt_lower / unique_sentences_lower)),
        
        'genders_nodata':           _get_col_total(_pt, NODATA),
        'genders_male':             _males,
        'genders_female':           _females,
        'genders_other':            _get_col_total(_pt, 'other'),

        'ages_nodata':              _get_row_total(_pt, NODATA),
        'ages_teens':               _get_row_total(_pt, 'teens'),
        'ages_twenties':            _get_row_total(_pt, 'twenties'),
        'ages_thirties':            _get_row_total(_pt, 'thirties'),
        'ages_fourties':            _get_row_total(_pt, 'fourties'),
        'ages_fifties':             _get_row_total(_pt, 'fifties'),
        'ages_sixties':             _get_row_total(_pt, 'sixties'),
        'ages_seventies':           _get_row_total(_pt, 'seventies'),
        'ages_eighties':            _get_row_total(_pt, 'eighties'),
        'ages_nineties':            _get_row_total(_pt, 'nineties'),

        'genders_f_m_ratio':        float("{:.4f}".format(_females / _males)) if _males > 0 else -1,

    }
    return results

#
# Main loop for experiments-versions-locales
#

def main() -> None:
    print('=== Diversity checker for Common Voice Datasets ===')
    print()

    # START TEST AREA
    # handle_split('cc0-default', 'cv-corpus-10.0-2022-07-04', 'tr', 'validated',
    #     'C:\\GITREPO\\_HK_GITHUB\\common-voice-diversity-check\\experiments\\cc0-default\\cv-corpus-10.0-2022-07-04\\tr\\validated.tsv'
    #     )
    # sys.exit(0)
    # END TEST AREA

    # TODO these should be set as arguments
    experiments_dir: str = os.path.join(HERE, 'experiments')
    # print(HERE)
    # print(experiments_dir)

    # Prepare final dataframe
    resdf: pd.DataFrame = pd.DataFrame(columns=RESULT_COLS).reset_index(drop=True)

    # Get total for progress display
    all_train: "list[str]" = glob.glob(os.path.join(experiments_dir, '**', 'train.tsv'), recursive=True)
    # print(all_validated)
    total_dataset_cnt: int = len(all_train) # * len(SPLITS_IDS)

    # Get experiment list
    experiment_paths: "list[str]" = glob.glob(os.path.join(experiments_dir, '*'), recursive=False)
    # print(experiments)

    # For each experiment
    cnt: int = 0 # counter
    for exp_path in experiment_paths:
        exp_name: str = os.path.split(exp_path)[-1]
        if VERBOSE:
            print(f'= Processing Experiment: {exp_name}')
        # Now get versions inside it
        exp_corpora_paths: "list[str]" = glob.glob(os.path.join(exp_path, '*'), recursive=False)
        # print(exp_corpora)

        # For each corpus
        for corpus_path in exp_corpora_paths:
            exp_corpus_name: str = os.path.split(corpus_path)[-1]
            if VERBOSE:
                print(f'== Processing Corpus: {exp_corpus_name}')
            # Now get the list of locales
            exp_corpus_locale_paths: "list[str]" = glob.glob(os.path.join(corpus_path, '*'), recursive=False)
            # print(exp_corpus_locales)
            
            # For each locale
            for locale_path in exp_corpus_locale_paths:
                cnt += 1
                locale_name: str = os.path.split(locale_path)[-1]
                if VERBOSE:
                    print(f'=== Processing Locale: {locale_name}')

                # Now get values for each split we are interested in
                for split_id in SPLITS_IDS:
                    split_path: str = os.path.join(locale_path, split_id + '.tsv')
                    if os.path.isfile(split_path):
                        if VERBOSE:
                            print(f'==== Processing Split: {split_path}')
                        else:
                            print('\033[F' + ' ' * 80)
                            print(f'\033[FProcessing {cnt}/{total_dataset_cnt} => {exp_name} - {exp_corpus_name} - {locale_name} - {split_id}')
                        # get stats
                        res = handle_split(exp=exp_name, ver=exp_corpus_name, lc=locale_name, split=split_id, fpath=split_path)
                        # add to resdf
                        resdf: pd.DataFrame = pd.concat([resdf.loc[:], pd.DataFrame([res])])
                # done splits in locale
            # done locales in version
        # done version in versions
    # done experiment in experiments

    # now, save the result
    fn: str = os.path.join(HERE, 'results', '$diversity_data.tsv')
    df_write(resdf, fn)



main()
