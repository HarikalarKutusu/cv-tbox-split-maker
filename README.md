# Common Voice Diversity Check

Simple script which checks diversity in Mozilla Common Voice default splits and/or alternative splits you provide for multiple CV versions and languages.

Currently only gender is processed.

To use:

- Clone the repo, modify the experiment directory for your tests.
- Create experiment directories under "experiments"
- Put your tsv files under version/language_code directories
- Run the "check-diversity" script

The script will scan all experiments, versions, and languages under experimenets directory and build a diversity.tsv file containing everything. You can then take it and analyze further.

So the directory structure is like:

```py
root
    experiment1
        cv-corpus-NN.N-YYYY-MM-DD
            locale1
            locale2
            ...
        ...
    experiment2
        ...
```

In this repo, we used a collection of Turkic Languages with v10.0 datasets as an example. These are the experiments in it:

- cc0-default: Default splits in datasets, created by the current Common Voice CorporaCreator
- cc0-s99: Alternative splits created by the current Common Voice CorporaCreator with -s 99 option, taking all recordings per sentence, i.e. the whole dataset.
