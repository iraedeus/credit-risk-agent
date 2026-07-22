from pathlib import Path

import pandas as pd
import pytest

from credit_risk_agent.data import StandardScaler


def test_fit_computes_correct_mean_and_std() -> None:
    """Verify that fit calculates mean and std correctly for multiple columns."""
    # Arrange
    df = pd.DataFrame({"f1": [10.0, 20.0, 30.0], "f2": [1.0, 2.0, 3.0]})
    columns = ["f1", "f2"]

    # Act
    scaler = StandardScaler().fit(df, columns)

    # Assert
    assert scaler.mean["f1"] == pytest.approx(20.0)
    assert scaler.mean["f2"] == pytest.approx(2.0)
    assert scaler.std["f1"] == pytest.approx(10.0)
    assert scaler.std["f2"] == pytest.approx(1.0)


def test_transform_normalizes_data_correctly() -> None:
    """Verify Z-score transformation on target columns while preserving non-target columns."""
    # Arrange
    df = pd.DataFrame({"f1": [10.0, 20.0, 30.0], "other": [100, 200, 300]})
    scaler = StandardScaler(mean={"f1": 20.0}, std={"f1": 10.0})

    # Act
    result = scaler.transform(df, ["f1"])

    # Assert
    assert result["f1"].tolist() == pytest.approx([-1.0, 0.0, 1.0])
    assert result["other"].tolist() == [100, 200, 300]


def test_fit_zero_std_handling() -> None:
    """Verify that constant features with zero std do not cause division by zero."""
    # Arrange
    df = pd.DataFrame({"f1": [5.0, 5.0, 5.0]})

    # Act
    scaler = StandardScaler().fit(df, ["f1"])
    result = scaler.transform(df, ["f1"])

    # Assert
    assert scaler.std["f1"] == 1.0
    assert result["f1"].tolist() == [0.0, 0.0, 0.0]


def test_transform_non_inplace_safety() -> None:
    """Verify that transform returns a new DataFrame and does not modify the original slice."""
    # Arrange
    df = pd.DataFrame({"f1": [10.0, 20.0, 30.0]})
    sliced_df = df.iloc[0:2]
    scaler = StandardScaler(mean={"f1": 20.0}, std={"f1": 10.0})

    # Act
    result = scaler.transform(sliced_df, ["f1"])

    # Assert
    assert sliced_df["f1"].tolist() == [10.0, 20.0]
    assert result["f1"].tolist() == pytest.approx([-1.0, 0.0])


def test_transform_missing_columns_raises_value_error() -> None:
    """Verify that ValueError is raised when trying to transform a column absent from scaler parameters."""
    # Arrange
    df = pd.DataFrame({"f1": [10.0, 20.0], "missing_col": [1.0, 2.0]})
    scaler = StandardScaler(mean={"f1": 20.0}, std={"f1": 10.0})

    # Act & Assert
    with pytest.raises(ValueError, match="absent in the scaler parameters"):
        scaler.transform(df, ["f1", "missing_col"])


def test_save_and_load_persistence(tmp_path: Path) -> None:
    """Verify JSON serialization and deserialization of StandardScaler parameters."""
    # Arrange
    df = pd.DataFrame({"f1": [10.0, 20.0, 30.0]})
    columns = ["f1"]
    save_path = tmp_path / "scaler.json"

    scaler = StandardScaler().fit(df, columns)

    # Act
    scaler.save(save_path)
    loaded_scaler = StandardScaler.load(save_path)
    result = loaded_scaler.transform(df, columns)

    # Assert
    assert save_path.exists()
    assert loaded_scaler.mean == scaler.mean
    assert loaded_scaler.std == scaler.std
    assert result["f1"].tolist() == pytest.approx([-1.0, 0.0, 1.0])
