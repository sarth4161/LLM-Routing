# make_splits.py - adds a `split` column (train/tune/test) to data/queries.csv if missing.
# 60/20/20, stratified by task_type + difficulty, fixed seed. Cache-test rows
# (those with a cache_group) and the originals they point to always go to test,
# so eval can ask the original first and then check if the variant hits the cache.
# Usage: python -m scripts.make_splits [--force]
import shutil
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from backend import config

SEED = 42


def main():
    df = pd.read_csv(config.QUERIES_CSV, dtype=str)
    if "split" in df.columns and "--force" not in sys.argv:
        print("queries.csv already has a split column:", df.split.value_counts().to_dict())
        return
    shutil.copy(config.QUERIES_CSV, config.QUERIES_CSV.with_suffix(".csv.bak"))

    has_group = df.cache_group.notna() & (df.cache_group != "")
    forced_test = set(df.loc[has_group, "id"]) | set(df.loc[has_group, "cache_group"])
    free = df[~df.id.isin(forced_test)]

    n_originals = (~has_group).sum()
    n_test_left = max(0, round(0.2 * n_originals) - len(set(df.loc[has_group, "cache_group"])))
    strata = free.task_type + "_" + free.difficulty
    rest, test = train_test_split(free, test_size=n_test_left, stratify=strata, random_state=SEED)
    train, tune = train_test_split(rest, test_size=round(0.2 * n_originals),
                                   stratify=rest.task_type + "_" + rest.difficulty,
                                   random_state=SEED)

    df["split"] = "test"
    df.loc[df.id.isin(train.id), "split"] = "train"
    df.loc[df.id.isin(tune.id), "split"] = "tune"
    df.to_csv(config.QUERIES_CSV, index=False)
    print("Wrote split column (backup: queries.csv.bak)")
    print(pd.crosstab(df.split, df.task_type, margins=True))


if __name__ == "__main__":
    main()
