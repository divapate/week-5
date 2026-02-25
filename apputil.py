# apputil.py

import pandas as pd
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv"


# -------------------------------------------------
# Exercise 1: Survival Demographics
# -------------------------------------------------

def survival_demographics():

    df = pd.read_csv(DATA_URL)

    bins = [0, 12, 19, 59, np.inf]
    labels = ["Child", "Teen", "Adult", "Senior"]

    # Create categorical age groups
    df["age_group"] = pd.cut(
        df["Age"],
        bins=bins,
        labels=labels
    )

    df["age_group"] = df["age_group"].astype(
        pd.CategoricalDtype(categories=labels)
    )

    # Create full index of all 24 combinations
    full_index = pd.MultiIndex.from_product(
        [
            sorted(df["Pclass"].unique()),
            sorted(df["Sex"].unique()),
            labels
        ],
        names=["Pclass", "Sex", "age_group"]
    )

    grouped = (
        df.groupby(["Pclass", "Sex", "age_group"])
        .agg(
            n_passengers=("Survived", "count"),
            n_survivors=("Survived", "sum")
        )
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    grouped["survival_rate"] = (
        grouped["n_survivors"] /
        grouped["n_passengers"]
    )

    return grouped


# -------------------------------------------------
# Exercise 2: Family Size and Wealth
# -------------------------------------------------

def family_groups():

    df = pd.read_csv(DATA_URL)

    df["family_size"] = df["SibSp"] + df["Parch"] + 1

    grouped = df.groupby(
        ["family_size", "Pclass"],
        as_index=False
    ).agg(
        n_passengers=("Fare", "count"),
        avg_fare=("Fare", "mean"),
        min_fare=("Fare", "min"),
        max_fare=("Fare", "max")
    )

    return grouped


def last_names():

    df = pd.read_csv(DATA_URL)

    # Return Series (required by grader)
    return (
        df["Name"]
        .str.split(",")
        .str[0]
        .value_counts()
    )


# -------------------------------------------------
# Bonus: Age Division
# -------------------------------------------------

def determine_age_division():

    median_age = df.groupby("Pclass")["Age"].transform("median")

    df["older_passenger"] = df["Age"] > median_age

    return df
