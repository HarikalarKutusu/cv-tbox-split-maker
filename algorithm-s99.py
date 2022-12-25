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

import corporacreator as cc

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
    corpus: pd.DataFrame = df_read(val_path)

    # Must have records in it
    if corpus.shape[0] > 0:
        # create temp dir
        os.makedirs(temp_path, exist_ok=True)

        # handle corpus
        args = cc.parse_args(
            ['-d', temp_path, '-f', val_path, '-s', str(duplicate_sentences)]
            )
        corpora: cc.Corpus = cc.Corpus(args, lc, corpus)
        corpora.create()
        corpora.save(temp_path)

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
