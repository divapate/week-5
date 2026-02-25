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

    df["age_group"] = pd.cut(
        df["Age"],
        bins=bins,
        labels=labels
    )

    grouped = df.groupby(
        ["Pclass", "Sex", "age_group"],
        as_index=False,
        observed=False  # ensures all 24 combinations appear
    ).agg(
        n_passengers=("Survived", "count"),
        n_survivors=("Survived", "sum")
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

    # Must return a pandas Series (NOT DataFrame)
    last_name_series = (
        df["Name"]
        .str.split(",")
        .str[0]
        .value_counts()
    )

    return last_name_series


# -------------------------------------------------
# Bonus: Age Division
# -------------------------------------------------

def determine_age_division():

    df = pd.read_csv(DATA_URL)

    median_age = df.groupby("Pclass")["Age"].transform("median")

    df["older_passenger"] = df["Age"] > median_age

    return df
