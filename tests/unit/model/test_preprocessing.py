import pandas as pd

from model.preprocessing import STATIC_FEATURES, preprocess, preprocess_static


def test_preprocess_marriage_mapping() -> None:
    # Arrange
    df = pd.DataFrame({"marriage": [0, 1, 2, 3], "education": [1, 2, 3, 4]})

    # Act
    result = preprocess(df)

    # Assert
    # 0 must be replaced by 3, other values remain intact
    assert result["marriage"].tolist() == [3, 1, 2, 3]


def test_preprocess_education_mapping() -> None:
    # Arrange
    df = pd.DataFrame({"marriage": [1, 1, 1, 1, 1], "education": [0, 2, 4, 5, 6]})

    # Act
    result = preprocess(df)

    # Assert
    # 0 must be replaced by 4
    # Values > 4 (5, 6) must be replaced by 4
    # Values <= 4 (2, 4) must remain intact
    assert result["education"].tolist() == [4, 2, 4, 4, 4]


def test_preprocess_non_inplace_safety() -> None:
    # Arrange
    df = pd.DataFrame({"marriage": [0, 1], "education": [0, 5]})

    # Act
    result = preprocess(df)

    # Assert
    # Original DataFrame must not be modified
    assert df["marriage"].tolist() == [0, 1]
    assert df["education"].tolist() == [0, 5]
    # Returned DataFrame is correctly preprocessed
    assert result["marriage"].tolist() == [3, 1]
    assert result["education"].tolist() == [4, 4]


def test_preprocess_static_ohe_and_alignment() -> None:
    # Arrange
    # Create a dataframe with partial features (some categories missing, sex=2 is missing, education=3,4 missing, etc.)
    df = pd.DataFrame(
        {
            "limit_bal": [10000.0, 20000.0],
            "sex": [1, 1],
            "marriage": [1, 2],
            "education": [1, 2],
            "age": [20, 60],  # age 20 (bin 0), age 60 (bin 3)
        }
    )

    # Act
    result = preprocess_static(df)

    # Assert
    # Check that shape is exactly (2, 14) since STATIC_FEATURES has 14 features
    assert result.shape == (2, 14)

    # Check order of columns matches STATIC_FEATURES
    assert list(result.columns) == STATIC_FEATURES

    # Check values for client 0: limit_bal=10000.0, sex=1, marriage=1, education=1, age_binned=0
    # Expected columns: limit_bal=10000.0, sex_1=1, sex_2=0, marriage_1=1, marriage_2=0, marriage_3=0
    # education_1=1, education_2=0, education_3=0, education_4=0
    # age_binned_0=1, age_binned_1=0, age_binned_2=0, age_binned_3=0
    client_0 = result.iloc[0].to_dict()
    assert client_0["limit_bal"] == 10000.0
    assert client_0["sex_1"] == 1.0
    assert client_0["sex_2"] == 0.0
    assert client_0["marriage_1"] == 1.0
    assert client_0["marriage_2"] == 0.0
    assert client_0["marriage_3"] == 0.0
    assert client_0["education_1"] == 1.0
    assert client_0["education_2"] == 0.0
    assert client_0["education_3"] == 0.0
    assert client_0["education_4"] == 0.0
    assert client_0["age_binned_0"] == 1.0
    assert client_0["age_binned_1"] == 0.0
    assert client_0["age_binned_2"] == 0.0
    assert client_0["age_binned_3"] == 0.0

    # Check non-inplace safety
    assert "age_binned" not in df.columns
