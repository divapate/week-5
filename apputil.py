# apputil.py

import pandas as pd
import numpy as np


# -------------------------------------------------
# Exercise 1
# -------------------------------------------------

def survival_demographics(df):

    bins = [0, 12, 19, 59, np.inf]
    labels = ["Child", "Teen", "Adult", "Senior"]

    # Create categorical age group
    df["age_group"] = pd.cut(
        df["age"],
        bins=bins,
        labels=labels
    )

    df["age_group"] = df["age_group"].astype(
        pd.CategoricalDtype(categories=labels)
    )

    # Create full 24-combination index
    full_index = pd.MultiIndex.from_product(
        [
            sorted(df["pclass"].unique()),
            sorted(df["sex"].unique()),
            labels
        ],
        names=["pclass", "sex", "age_group"]
    )

    grouped = (
        df.groupby(["pclass", "sex", "age_group"])
        .agg(
            n_passengers=("survived", "count"),
            n_survivors=("survived", "sum")
        )
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    grouped["survival_rate"] = (
        grouped["n_survivors"] /
        grouped["n_passengers"]
    )

    # Ensure categorical dtype survives reset_index
    grouped["age_group"] = grouped["age_group"].astype(
        pd.CategoricalDtype(categories=labels)
    )

    return grouped


# -------------------------------------------------
# Exercise 2
# -------------------------------------------------

def family_groups(df):

    df["family_size"] = df["sibsp"] + df["parch"] + 1

    grouped = df.groupby(
        ["family_size", "pclass"],
        as_index=False
    ).agg(
        n_passengers=("fare", "count"),
        avg_fare=("fare", "mean"),
        min_fare=("fare", "min"),
        max_fare=("fare", "max")
    )

    return grouped


def last_names(df):

    return (
        df["name"]
        .str.split(",")
        .str[0]
        .value_counts()
    )


# -------------------------------------------------
# Bonus
# -------------------------------------------------

def determine_age_division(df):

    median_age = df.groupby("pclass")["age"].transform("median")

    df["older_passenger"] = df["age"] > median_age

    return df
