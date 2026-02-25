import pandas as pd
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv"

# -------------------------------------------------
# Exercise 1: Survival Demographics
# -------------------------------------------------

def survival_demographics():
    # Load data and force lowercase column names
    df = pd.read_csv(DATA_URL)
    df.columns = df.columns.str.lower()

    bins = [0, 12, 19, 59, np.inf]
    labels = ["Child", "Teen", "Adult", "Senior"]

    # Create categorical age groups (pd.cut returns Categorical when labels given)
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)

    # Build full index using the actual categories from the column
    full_index = pd.MultiIndex.from_product(
        [
            sorted(df["pclass"].unique()),
            sorted(df["sex"].unique()),
            df["age_group"].dtype.categories  # preserves categorical nature
        ],
        names=["pclass", "sex", "age_group"]
    )

    # Group with observed=True to avoid FutureWarning
    grouped = (
        df.groupby(["pclass", "sex", "age_group"], observed=True)
        .agg(
            n_passengers=("survived", "count"),
            n_survivors=("survived", "sum")
        )
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    # Ensure age_group remains categorical (reset_index converts to object)
    grouped["age_group"] = grouped["age_group"].astype(
        pd.CategoricalDtype(categories=labels)
    )

    grouped["survival_rate"] = grouped["n_survivors"] / grouped["n_passengers"]
    return grouped


# -------------------------------------------------
# Exercise 2: Family Size and Wealth
# -------------------------------------------------

def family_groups():
    df = pd.read_csv(DATA_URL)
    df.columns = df.columns.str.lower()

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


def last_names():
    df = pd.read_csv(DATA_URL)
    df.columns = df.columns.str.lower()

    # Return Series of last name counts (as required by grader)
    return df["name"].str.split(",").str[0].value_counts()


# -------------------------------------------------
# Bonus: Age Division
# -------------------------------------------------

def determine_age_division():
    # Load data and force lowercase column names
    df = pd.read_csv(DATA_URL)
    df.columns = df.columns.str.lower()

    # Calculate median age per passenger class
    median_age = df.groupby("pclass")["age"].transform("median")

    # Create boolean column, but set NaN where age is missing
    df["older_passenger"] = (df["age"] > median_age).where(df["age"].notna())

    return df
