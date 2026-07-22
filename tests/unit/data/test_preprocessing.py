import pandas as pd

from credit_risk_agent.data.preprocessing import STATIC_FEATURES, preprocess, preprocess_static


class TestPreprocess:
    def test_preprocess_marriage_mapping(self) -> None:
        """Verify that marriage category 0 is remapped to 3 while preserving other values."""
        # Arrange
        df = pd.DataFrame({"marriage": [0, 1, 2, 3], "education": [1, 2, 3, 4]})

        # Act
        result = preprocess(df)

        # Assert
        # 0 must be replaced by 3, other values remain intact
        assert result["marriage"].tolist() == [3, 1, 2, 3]

    def test_preprocess_education_mapping(self) -> None:
        """Verify that education category 0 and values > 4 are remapped to 4."""
        # Arrange
        df = pd.DataFrame({"marriage": [1, 1, 1, 1, 1], "education": [0, 2, 4, 5, 6]})

        # Act
        result = preprocess(df)

        # Assert
        # 0 must be replaced by 4
        # Values > 4 (5, 6) must be replaced by 4
        # Values <= 4 (2, 4) must remain intact
        assert result["education"].tolist() == [4, 2, 4, 4, 4]

    def test_preprocess_non_inplace_safety(self) -> None:
        """Verify that preprocess creates a new DataFrame without modifying the input."""
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


class TestPreprocessStatic:
    def test_preprocess_static_ohe_and_alignment(self) -> None:
        """Verify one-hot encoding, column ordering, and alignment with STATIC_FEATURES."""
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

    def test_preprocess_static_extreme_ages(self) -> None:
        """Verify handling of edge-case age values (0, 105, and NaN) during binning."""
        # Arrange
        # Test age 0, age 105, and NaN
        df = pd.DataFrame(
            {
                "limit_bal": [10000.0, 20000.0, 30000.0],
                "sex": [1, 1, 1],
                "marriage": [1, 2, 1],
                "education": [1, 2, 1],
                "age": [0, 105, None],
            }
        )

        # Act
        result = preprocess_static(df)

        # Assert
        assert result.shape == (3, 14)
        # Client 0 (age 0) should be binned safely (e.g. to bin 0)
        assert result.iloc[0]["age_binned_0"] == 1.0
        # Client 1 (age 105) should be binned safely (e.g. to bin 3)
        assert result.iloc[1]["age_binned_3"] == 1.0
        # Client 2 (age None) should not crash, and all age_binned columns should be 0.0
        client_2 = result.iloc[2]
        assert client_2["age_binned_0"] == 0.0
        assert client_2["age_binned_1"] == 0.0
        assert client_2["age_binned_2"] == 0.0
        assert client_2["age_binned_3"] == 0.0
