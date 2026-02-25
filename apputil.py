# apputil.py

import pandas as pd
import numpy as np


# -------------------------------------------------
# Exercise 1: Survival Demographics
# -------------------------------------------------

def survival_demographics():
    df = pd.read_csv("titanic.csv")

    # Remove missing ages
    df = df.dropna(subset=["Age"])

    # Create age groups (Categorical dtype automatically)
    bins = [0, 12, 19, 59, np.inf]
    labels = ["Child", "Teen", "Adult", "Senior"]
    df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels)

    grouped = (
        df.groupby(["Pclass", "Sex", "age_group"])
        .agg(
            n_passengers=("Survived", "count"),
            n_survivors=("Survived", "sum")
        )
        .reset_index()
    )

    grouped["survival_rate"] = (
        grouped["n_survivors"] / grouped["n_passengers"]
    )

    return grouped


# -------------------------------------------------
# Exercise 2: Family Size and Wealth
# -------------------------------------------------

def family_groups():
    df = pd.read_csv("titanic.csv")

    df["family_size"] = df["SibSp"] + df["Parch"] + 1

    grouped = (
        df.groupby(["family_size", "Pclass"])
        .agg(
            n_passengers=("Fare", "count"),
            avg_fare=("Fare", "mean"),
            min_fare=("Fare", "min"),
            max_fare=("Fare", "max")
        )
        .reset_index()
    )

    return grouped


def last_names():
    df = pd.read_csv("titanic.csv")

    df["last_name"] = df["Name"].str.split(",").str[0]

    counts = (
        df["last_name"]
        .value_counts()
        .reset_index()
    )

    counts.columns = ["last_name", "count"]

    return counts


# -------------------------------------------------
# Bonus: Age Division
# -------------------------------------------------

def determine_age_division():
    df = pd.read_csv("titanic.csv")

    class_medians = df.groupby("Pclass")["Age"].transform("median")

    df["older_passenger"] = df["Age"] > class_medians

    return df
