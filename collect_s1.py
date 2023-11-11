#!/usr/bin/env python3

"""
###########################################################################
# collect_s1.py
#
# Copy the *.tsv files from CV release to experiments/s1/[release]
#
# Use:
# python collect_s1.py
#
# This script is part of Common Voice ToolBox Package
#
# [github]
# [copyright]
###########################################################################
"""

# Standard Lib
import os
import sys
import glob
import shutil
from datetime import datetime

# External Dependencies
import progressbar

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)


def main() -> None:
    """Main process"""
    release_name: str = os.path.split(args_src_dir)[-1]
    src_locale_dirs: list[str] = glob.glob(os.path.join(args_src_dir, "*"))
    total_cnt: int = len(src_locale_dirs)
    print(f"Copying .TSV files from {total_cnt} locales")
    bar: progressbar.ProgressBar = progressbar.ProgressBar(max_value=total_cnt)
    cnt: int = 0
    start_time: datetime = datetime.now()
    bar.start()
    for src_lc_dir in src_locale_dirs:
        bar.update(cnt)
        cnt += 1
        if os.path.isdir(src_lc_dir):
            lc: str = os.path.split(src_lc_dir)[-1]
            dst_dir: str = os.path.join(HERE, "experiments", "s1", release_name, lc)
            os.makedirs(dst_dir, exist_ok=True)
            files: list[str] = glob.glob(os.path.join(src_lc_dir, "*.tsv"))
            for f in files:
                shutil.copy(f, dst_dir)
    # finalize
    bar.finish()
    end_time: datetime = datetime.now()
    seconds: int = (end_time - start_time).seconds
    print(f"Finished in {seconds} sec - Avg={seconds/total_cnt} sec/locale")


if __name__ == "__main__":
    # [TODO] : use command line args
    args_src_dir: str = "m:\\DATASETS\\cv\\cv-corpus-15.0-2023-09-08"

    main()
