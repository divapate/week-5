# apputil.py

import pandas as pd
import numpy as np


# -------------------------------------------------
# Exercise 1: Survival Demographics
# -------------------------------------------------

def survival_demographics(df):

    bins = [0, 12, 19, 59, np.inf]
    labels = ["Child", "Teen", "Adult", "Senior"]

    # Create age_group as categorical
    df["age_group"] = pd.cut(
        df["Age"],
        bins=bins,
        labels=labels
    )

    df["age_group"] = df["age_group"].astype(
        pd.CategoricalDtype(categories=labels)
    )

    # Group and ensure all 24 combinations appear
    grouped = (
        df.groupby(["Pclass", "Sex", "age_group"], observed=False)
        .agg(
            n_passengers=("Survived", "count"),
            n_survivors=("Survived", "sum")
        )
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

def family_groups(df):

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


def last_names(df):

    # Must return a pandas Series
    return (
        df["Name"]
        .str.split(",")
        .str[0]
        .value_counts()
    )


# -------------------------------------------------
# Bonus: Age Division
# -------------------------------------------------

def determine_age_division(df):

    median_age = df.groupby("Pclass")["Age"].transform("median")

    df["older_passenger"] = df["Age"] > median_age

    return df

