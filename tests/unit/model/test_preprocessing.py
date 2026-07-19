import pandas as pd

from model.preprocessing import preprocess


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
