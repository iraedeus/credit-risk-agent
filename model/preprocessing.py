import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[df["marriage"] == 0, "marriage"] = 3
    df.loc[df["education"] == 0, "education"] = 4
    df.loc[df["education"] > 4, "education"] = 4

    return df
