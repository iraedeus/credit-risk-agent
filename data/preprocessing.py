import pandas as pd

STATIC_FEATURES = [
    "limit_bal",
    "sex_1",
    "sex_2",
    "marriage_1",
    "marriage_2",
    "marriage_3",
    "education_1",
    "education_2",
    "education_3",
    "education_4",
    "age_binned_0",
    "age_binned_1",
    "age_binned_2",
    "age_binned_3",
]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input credit risk DataFrame by cleaning categorical columns.

    This function copies the input DataFrame and maps undefined categorical values to
    known baseline categories:
    - Sets marriage status code 0 (which is undocumented) to 3 (others).
    - Sets education code 0 to 4 (others).
    - Sets education codes greater than 4 to 4 (others).

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing raw client features (must contain 'marriage'
        and 'education' columns).

    Returns
    -------
    pd.DataFrame
        A new DataFrame with cleaned categorical columns.
    """

    df = df.copy()
    df.loc[df["marriage"] == 0, "marriage"] = 3
    df.loc[df["education"] == 0, "education"] = 4
    df.loc[df["education"] > 4, "education"] = 4

    return df


def preprocess_static(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering and One-Hot Encoding on static client characteristics.

    This function processes static columns (age, sex, marriage, education, limit_bal):
    - Bins the age column into 4 age groups (age_binned).
    - One-hot encodes the categorical features: sex, marriage, education, and age_binned.
    - Standardizes the column structure by reindexing to STATIC_FEATURES, ensuring
      a deterministic output shape and column order, filled with zeros for missing columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing client records (must contain 'age', 'sex', 'marriage',
        'education', and 'limit_bal' columns).

    Returns
    -------
    pd.DataFrame
        One-hot encoded and aligned DataFrame containing only STATIC_FEATURES columns.
    """
    df = df.copy()
    df["age_binned"] = pd.cut(df["age"], bins=[-float("inf"), 25, 35, 50, float("inf")], labels=[0, 1, 2, 3])

    df_ohe = pd.get_dummies(df, columns=["sex", "marriage", "education", "age_binned"], dtype=float)

    df_ohe = df_ohe.reindex(columns=STATIC_FEATURES, fill_value=0.0)

    return df_ohe
