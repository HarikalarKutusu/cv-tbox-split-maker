# Common Voice Diversity Check

A collection of scripts to create alternative splits, to check important measures in multiple Common Voice relases, languages and alternate splitting strategies.

This tooling will be part of Common Voice ToolBox, released separately. It will evantually be transformed into a more generalized script in the core.

## Why?

- Before feeding your dataset into a lengthy training it is a must to check the health of your dataset splits. Do you train with male voices only? Are there enough distinct voices in train/dev/test sets? How much of your validated data does your actual splits use?
- You are a Common Voice language community lead, a new release comes out, you did a great job in your last campaign! Or did you? Did your female/male ratio improve as you aimed?
- You work on multiple Common Voice languages and want to compare them.
- How is your language dataset changing through versions/years?
- You want to use different alternative splitting strategies and want to compare it with the default splits or other strategies, but are they diverse enough?
- Or, all of the above...

The process of doing these with L languages, V versions and S splitting strategies (+three splitting algorithms in each) means repeated processing of 3*L*V\*S splits.

This is where this tool comes in. You just put your data in directories and feed them to scripts. The basic measures are compiled into a tsv file which you can process with your own scripts or other tools. We also included an MS Excel file as an analysis frontend to this data.

## Scripts

In the required execution order:

- **extract.py** : Expands .tsv files (or all files with `--all` option) from downloaded .tar.gz dataset files in a directory, into another directory. If `--all` is not specified, audio files under `clips` are not expanded.
- **collect_s1.py** : From the expanded datasets, copies necessary files to internal space to work on (these include common .tsv files and default split files).
- **algorithm_xxx.py** : Different splitting algorithms to execute (see below)
- **tbox_diversity_table.py** : The script will scan all experiments, versions, and languages under experiments directory and builds results/$diversity_data.tsv file containing everything. You can then take it and analyze further or use the Excel file provided.

## How

To prepare:

- Clone the repo and cd into it
- Use `Python v3.11+`, create a venv and activate it.
- Run `pip install -U -r requirements.txt` to install the dependencies.
- Put your downloaded dataset(s) into a directory (e.g. /downloaded_datasets/cv-corpus-16.1-2023-12-06)
- Create a directory for expanded dataset files (e.g. /datasets)
- Edit `config.py` to point to them
- Run `python extract.py` to extract only the .tsv files from the datasets
- Run `python collect.py` to copy the metadata files into the internal working area
- Run the aplitting algoritm(s) you desire

The internal data directory structure is like:

```py
clone_root/experiments
    <exp>                           # e.g. s1, s99, v1
        <cv-corpus>                 # e.g. cv-corpus-NN.N-YYYY-MM-DD
            <lc>                    # Language directory (en, tr etc)
                *.tsv               # Metadata (*.tsv only)
            ...
        ...
    <exp>
        ...
```

Under `experiments/s1`, all .tsv files from the release can be found. Other algorithm directories only contain train/dev/test.tsv files.

## Algorithms and the data

The data we use is huge and not suited for github. We used the following:

- s1: Default splits in datasets, created by the current Common Voice CorporaCreator (-s 1 default option)
- s99: Alternative splits created by the current Common Voice CorporaCreator with -s 99 option, taking up to 99 recordings per sentence, i.e. nearly the whole dataset.
- v1: ("Voice First") Alternative split created by "algorithm-v1.py". This is a 80-10-10% (train-dev-test) splitting with 25-25-50% voice diversity (nearly) ensured. This algorithm has been suggested to Common Voice Project for being used as default splitting algorithm, with a report including a detailed analysis and some training results.
- vw: ("Voice first for Whisper") A version of v1 for better OpenAI Whisper fine-tuning, with 90-5-5% splits, keeping 25-25-50% diversity (only available for Whisper languages).
- vx: ("Voice first for eXternal test") A version of v1 with 95-5-0% splits and 50-50-0% diversity, so no test split, where you can test your model against other datasets like Fleurs or Voxpopuli (only available for Fleurs & Voxpopuli languages).

Compressed splits for each language / dataset version / algorithm can be found under the [shared Google Drive location](https://drive.google.com/drive/folders/13c3VME_qRT1JSGjPue153K8FiDBH4QD2?usp=drive_link). To use them in your trainings, just download one and override the default train/dev/test.tsv files in your expanded dataset directory.

## Other

### License

AGPL v3.0

### TO-DO/Project plan, issues and feature requests

You can look at the results [Common Voice Dataset Analyzer](https://github.com/HarikalarKutusu/cv-tbox-dataset-analyzer).
This will eventually be part of the Common Voice Toolbox's Core, but it will be developed here...

The project status can be found on the [project page](https://github.com/users/HarikalarKutusu/projects/10). Please post [issues and feature requests](https://github.com/HarikalarKutusu/common-voice-diversity-check/issues), or [Pull Requests](https://github.com/HarikalarKutusu/common-voice-diversity-check/pulls) to enhance.

---

---

## FUTURE REFERENCE

Although this will go into the core, we will also publish it separately. Below you'll find what it will become (tentative, might change):

### tbox_split_maker

The script can use alternative splitting strategies for you to try on your language/languages.
Then you should re-run "tbox_diversity_table" to analyze statistics in these new splits and compare with other strategies.

```cmd
python3 tbox_split_maker <--split_strategy|--ss <strategy_code> [<parameter>] > [--exp <experiments_directory>] --in <path|experiment> --out <path|experiment> [--verbose]
```

**Options:**

--exp <experiments_directory> : If given, experiments are searched/created under this directory. If not given, experiments directory under cloned repo will be used.

--in <path|experiment> : If an existing full path is given, that directory is used to feed the default splits (eg: --in c:\datasets\cv\v10.0\en). If only a string is given (e.g. --in releases) given string is assumed to be under the experiments_directory and searched there.

--out <path|experiment> : If an existing full path is given (e.g. --out d:\trials\splits), it will be used for output of new splits. If only a string is given (e.g. --out releases) given string is assumed to be under the experiments_directory and created there. In either case, first the source is copied to destination and THEN the new slits override the existing ones. Other unrelated .tsv files are also being copied for dataset completeness. If your splits seem ok, you can just copy-override the original dataset you downladed and expanded with these files.000

--verbose: Prints out more information, by default minimal information is displayed.

**Currently supported strategies:**

--split_strategy cc [N]

Run Common Voice's Corpora Creator with alternative recordings per sentence setting. By default, this is 1, meaning there is 1 recording per sentence in the final splits, even if different users might record sentences multiple times. Although this setting is meaningful to prevent sentence bias, it might be desirable to have sentences recorded by different voices/genders/ages/accents so that your model gets better on alternatives. Also, especially with low resource languages, the default setting drops the training split size to a small fraction of what's available.

For this to work, you need to clone and compile Mozilla Common Voice CorporaCreator repo as follows:

```sh
git clone https://github.com/common-voice/CorporaCreator.git
cd CorporaCreator; python3 setup.py install
```

--split_strategy sentence <train_percent> <dev_percent> <test_percent>

In this strategy sentence unbiasing has the presedence, so that no sentence is repeated in other splits. But voices (people) can exists in other splits. This strategy ensures that the whole validated set is used. You might like to experiment with percentages thou. Usually 80-10-10 or 70-15-15 are considered good values (make sure they add up to 100).

Ex:

```sh
python tbox_split_maker --exp ~/cv/experiments --in default --out test-70-15-15 --ss sentence 70 15 15
```

--split_strategy sentence-voice <train_percent> <dev_percent> <test_percent>

This algorithm is similar to the "sentence" strategy, but it ensures no same voice exists in other splits. Therefore the dataset will not fully be used. The test and dev will be as desired, but the train split will be smaller, not adding to 100. This strategy prevents both sentence and voice bias and usually uses most of the validated set. Unused amount totaly depends on the dataset, how much text-corpus, voice contributors, repeated recording it has, and if people are recording too few and/or too much sentences. Therefore it is a good practice to analyze the generated split with tbox_directory_table script...

Ex:

```sh
python tbox_split_maker --exp ~/cv/experiments --in default --out test-70-15-15 --ss sentence-voice 70 15 15 # note that these numbers are target, result will be different
```

--split_strategy random <train_percent> <dev_percent> <test_percent>

This is a dummy algorithm and does not care on any bias. It splits the whole dataset randomly and fullt. The reultant model performance in terms of bias might also be random. Here, also the split percentages doesn't need to add up to 100. This is provided for experimenting, by slicing different smaller sizes.

Ex:

```sh
python tbox_split_maker --exp ~/cv/experiments --in default --out random-50-10-10 --ss sentence-voice 50 10 10
```
