import pandas as pd


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
