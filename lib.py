""" cv-tbox Split Creator - Library """

# Standard Lib
import os
import sys
import csv
from argparse import Namespace
from datetime import datetime

# External Dependencies
import pandas as pd

# Module
import conf
from typedef import Globals

#
# Tooling
#


def dec2(x: float) -> str:
    """Return formatted float string to 2 decimal digits"""
    return str(round(x, 2))


def dec3(x: float) -> str:
    """Return formatted float string to 3 decimal digits"""
    return str(round(x, 3))


def final_report(g: Globals) -> None:
    """Reports the result statistics"""
    finish_time: datetime = datetime.now()
    process_seconds: float = (finish_time - g.start_time).total_seconds()
    avg_seconds: float = process_seconds / g.total_cnt if g.total_cnt > 0 else -1
    avg_seconds_real: float = (
        process_seconds / g.processed_cnt if g.processed_cnt > 0 else -1
    )
    print("\n" + "-" * 80)
    print(f"Total corpora\t\t: {g.total_cnt}")
    print(f"Source corpora\t\t: {g.src_cnt}")
    print(f"Skipped existing\t: {g.skipped_exists}")
    print(f"Skipped no support\t: {g.skipped_nosupport}")
    print(f"Skipped small\t\t: {g.skipped_small}")
    print(f"Skipped nodata\t\t: {g.skipped_nodata}")
    print(f"Actual Processed\t: {g.processed_cnt}")
    print(f"Total duration\t\t: {dec3(process_seconds)} sec")
    print(f"Avg. duration\t\t: {dec3(avg_seconds)} sec")
    print(f"Real average\t\t: {dec3(avg_seconds_real)} sec")
    print("-" * 80)


#
# DataFrame file read-write
#


def df_read(fpath: str) -> pd.DataFrame:
    """Read a tsv file into a dataframe"""
    if not os.path.isfile(fpath):
        print(f"FATAL: File {fpath} cannot be located!")
        if conf.FAIL_ON_NOT_FOUND:
            sys.exit(1)

    df: pd.DataFrame = pd.read_csv(
        fpath,
        sep="\t",
        parse_dates=False,
        engine="python",
        encoding="utf-8",
        on_bad_lines="skip",
        quotechar='"',
        quoting=csv.QUOTE_NONE,
    )
    return df


def df_write(df: pd.DataFrame, fpath: str) -> None:
    """Write dataframe to a tsv file"""
    df.to_csv(
        fpath,
        header=True,
        index=False,
        encoding="utf-8",
        sep="\t",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        lineterminator="\n",
    )


#
# Remove deleted users
#
def remove_deleted_users(df_val: pd.DataFrame) -> pd.DataFrame:
    """Given validated, remove recordings of deleted users from it, before splitting"""
    if df_val.shape[0] == 0:
        return df_val
    deleted_set: set[str] = set(
        df_read(os.path.join(".", "data", "deleted_users.tsv"))
        .astype(str)["client_id"]
        .to_list()
    )
    return df_val[~df_val["client_id"].isin(deleted_set)]


#
# Adapted from original Corpora Creator - removed unneeded features
# - Removed logger
# - No need to re-partition (we already have validated)
# - No need to preprocess (s1 already preprocessed the data)
# - Create only train, dev, test
#


class LocalCorpus:
    """Corpus representing a Common Voice datasets for a given locale.
    Args:
      args ([str]): Command line parameters as list of strings
      locale (str): Locale this :class:`corporacreator.Corpus` represents
      validated (:class:`pandas.DataFrame`): `pandas.DataFrame` Containing the validated corpus data
    Attributes:
        args ([str]): Command line parameters as list of strings
        locale (str): Locale of this :class:`corporacreator.Corpus`
        validated (:class:`pandas.DataFrame`): `pandas.DataFrame` Containing the validated corpus data
    """

    # __args
    # __locale: str = ""
    # __corpus_data = []
    # __train: pd.DataFrame = pd.DataFrame()
    # __dev: pd.DataFrame = pd.DataFrame()
    # __test: pd.DataFrame = pd.DataFrame()
    # validated: pd.DataFrame = pd.DataFrame()

    def __init__(self, args: Namespace, locale: str, validated: pd.DataFrame) -> None:
        self.__args: Namespace = args
        self.__locale: str = locale
        self.validated: pd.DataFrame = validated
        self.train: pd.DataFrame = pd.DataFrame(columns=validated.columns)
        self.dev: pd.DataFrame = pd.DataFrame(columns=validated.columns)
        self.test: pd.DataFrame = pd.DataFrame(columns=validated.columns)

    def create(self) -> None:
        """Creates a :class:`corporacreator.Corpus` for `self.locale`."""
        self._post_process_valid_data()

    def _post_process_valid_data(self) -> None:
        # Remove duplicate sentences while maintaining maximal user diversity at the frame's start
        # [TODO]: Make addition of user_sentence_count cleaner
        speaker_counts: pd.DataFrame = pd.DataFrame(
            self.validated["client_id"].value_counts()
        ).reset_index()
        speaker_counts.columns = ["client_id", "user_sentence_count"]
        self.validated = self.validated.join(
            speaker_counts.set_index("client_id"), on="client_id"
        )
        self.validated = self.validated.sort_values(
            ["user_sentence_count", "client_id"]
        )

        # Get a subset here, depending on duplicate sentence count

        validated: pd.DataFrame = self.validated.groupby("sentence").head(
            self.__args.duplicate_sentence_count
        )

        validated = validated.sort_values(
            ["user_sentence_count", "client_id"], ascending=False
        )
        validated = validated.drop(columns="user_sentence_count")
        self.validated = self.validated.drop(columns="user_sentence_count")

        train: pd.DataFrame = pd.DataFrame(columns=validated.columns)
        dev: pd.DataFrame = pd.DataFrame(columns=validated.columns)
        test: pd.DataFrame = pd.DataFrame(columns=validated.columns)

        train_size: int = 0
        dev_size: int = 0
        test_size: int = 0

        if len(validated) > 0:
            # Determine train, dev, and test sizes
            train_size, dev_size, test_size = self._calculate_data_set_sizes(
                len(validated)
            )
            # Split into train, dev, and test datasets
            continous_client_index, uniques = pd.factorize(validated["client_id"])
            validated["continous_client_index"] = continous_client_index

            for i in range(max(continous_client_index), -1, -1):
                if (
                    len(test) + len(validated[validated["continous_client_index"] == i])
                    <= test_size
                ):
                    test = pd.concat(
                        [test, validated[validated["continous_client_index"] == i]],
                        sort=False,
                    )
                elif (
                    len(dev) + len(validated[validated["continous_client_index"] == i])
                    <= dev_size
                ):
                    dev = pd.concat(
                        [dev, validated[validated["continous_client_index"] == i]],
                        sort=False,
                    )
                else:
                    train = pd.concat(
                        [train, validated[validated["continous_client_index"] == i]],
                        sort=False,
                    )

        self.train = train.drop(columns="continous_client_index", errors="ignore")
        self.dev = dev.drop(columns="continous_client_index", errors="ignore")
        self.test = test[:train_size].drop(
            columns="continous_client_index", errors="ignore"
        )

    def _calculate_data_set_sizes(self, total_size):
        # Find maximum size for the training data set in accord with sample theory
        train_size: int = total_size
        dev_size: int = 0
        test_size: int = 0
        for train_size in range(total_size, 0, -1):
            # calculated_sample_size = int(corporacreator.sample_size(train_size))
            calculated_sample_size = int(calc_sample_size(train_size))
            if 2 * calculated_sample_size + train_size <= total_size:
                dev_size = calculated_sample_size
                test_size = calculated_sample_size
                break
        return train_size, dev_size, test_size

    def save(self, save_dir: str) -> None:
        """Saves this :class:`corporacreator.Corpus` in `directory`.
        Args:
          directory (str): Directory into which this `corporacreator.Corpus` is saved.
        """
        directory: str = os.path.join(save_dir, self.__locale)
        if not os.path.exists(directory):
            os.mkdir(directory)
        datasets: list[str] = ["train", "dev", "test"]

        # _logger.debug("Saving %s corpora..." % self.locale)
        for dataset in datasets:
            self._save(directory, dataset)
        # _logger.debug("Saved %s corpora." % self.locale)

    def _save(self, directory, dataset) -> None:
        path: str = os.path.join(directory, dataset + ".tsv")

        dataframe: pd.DataFrame = getattr(self, dataset)
        dataframe.to_csv(
            path,
            sep="\t",
            header=True,
            index=False,
            encoding="utf-8",
            escapechar="\\",
            quoting=csv.QUOTE_NONE,
        )


#
# Sample Size Calculation, taken from CorporaCreatOr repo (statistics.py)
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
    numerator: float = (z_score**2 * fraction_picking * (1 - fraction_picking)) / (
        margin_of_error**2
    )
    denominator: float = 1 + (
        z_score**2 * fraction_picking * (1 - fraction_picking)
    ) / (margin_of_error**2 * population_size)
    return numerator / denominator
