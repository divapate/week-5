# apputil.py

import pandas as pd
import numpy as np
import plotly.express as px


# -------------------------------------------------------
# Exercise 1: Survival Demographics
# -------------------------------------------------------

def survival_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates age groups and calculates survival statistics
    grouped by class, sex, and age group.
    """
    df = df.copy()

    # Drop missing ages to avoid empty bins
    df = df.dropna(subset=["Age"])

    # Create age groups
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

    grouped = grouped.sort_values(
        by=["Pclass", "Sex", "age_group"]
    )

    return grouped


def visualize_demographic(demo_df: pd.DataFrame):
    """
    Visualizes survival rate by age group, sex, and class.
    """
    fig = px.bar(
        demo_df,
        x="age_group",
        y="survival_rate",
        color="Sex",
        facet_col="Pclass",
        barmode="group",
        title="Survival Rate by Class, Sex and Age Group"
    )

    fig.update_layout(yaxis_title="Survival Rate")

    return fig


# -------------------------------------------------------
# Exercise 2: Family Size and Wealth
# -------------------------------------------------------

def family_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates fare statistics grouped by family size and class.
    """
    df = df.copy()

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

    grouped = grouped.sort_values(by=["family_size", "Pclass"])

    return grouped


def last_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts and counts passenger last names.
    """
    df = df.copy()

    df["last_name"] = df["Name"].str.split(",").str[0]

    counts = (
        df["last_name"]
        .value_counts()
        .reset_index()
    )

    counts.columns = ["last_name", "count"]

    return counts


def visualize_families(df: pd.DataFrame):
    """
    Visualizes relationship between family size and fare.
    """
    df = df.copy()
    df["family_size"] = df["SibSp"] + df["Parch"] + 1

    fig = px.scatter(
        df,
        x="family_size",
        y="Fare",
        color="Pclass",
        opacity=0.7,
        title="Family Size vs Fare by Passenger Class"
    )

    return fig


# -------------------------------------------------------
# Bonus: Age Division
# -------------------------------------------------------

def determine_age_division(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a column indicating whether a passenger is older
    than the median age of their passenger class.
    """
    df = df.copy()

    class_medians = df.groupby("Pclass")["Age"].transform("median")

    df["older_passenger"] = df["Age"] > class_medians

    return df


def visualize_age_division(df: pd.DataFrame):
    """
    Visualizes survival based on age relative to class median.
    """
    fig = px.histogram(
        df,
        x="older_passenger",
        color="Survived",
        barmode="group",
        title="Survival by Age Relative to Class Median"
    )

    return fig
