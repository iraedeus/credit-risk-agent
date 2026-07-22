import pandas as pd
import pytest

from credit_risk_agent.data.downloader import wide_to_long


def test_wide_to_long_basic() -> None:
    # Arrange
    data = {
        "ID": [1, 2],
        "PAY_1": [2, -1],
        "PAY_2": [3, 0],
        "OTHER_COL": ["A", "B"],
    }
    df = pd.DataFrame(data)

    # Act
    result = wide_to_long(df, "PAY_")

    # Assert
    # Output columns should be ['ID', 'month', 'PAY_']
    assert list(result.columns) == ["ID", "month", "PAY_"]

    # Month parsing check (PAY_1 -> 1, PAY_2 -> 2)
    assert sorted(result["month"].unique()) == [1, 2]

    # Data transformation check
    # For ID=1: PAY_1=2, PAY_2=3
    # For ID=2: PAY_1=-1, PAY_2=0
    row_1_1 = result[(result["ID"] == 1) & (result["month"] == 1)].iloc[0]
    row_1_2 = result[(result["ID"] == 1) & (result["month"] == 2)].iloc[0]
    row_2_1 = result[(result["ID"] == 2) & (result["month"] == 1)].iloc[0]
    row_2_2 = result[(result["ID"] == 2) & (result["month"] == 2)].iloc[0]

    assert row_1_1["PAY_"] == 2
    assert row_1_2["PAY_"] == 3
    assert row_2_1["PAY_"] == -1
    assert row_2_2["PAY_"] == 0


def test_wide_to_long_ignores_non_matching_columns() -> None:
    # Arrange
    data = {
        "ID": [1],
        "PAY_1": [2],
        "PAY_AMT1": [100],  # "PAY_" matches start, but replace("PAY_", "") -> "AMT1" is not digit
        "BILL_AMT2": [500],
    }
    df = pd.DataFrame(data)

    # Act
    result_pay = wide_to_long(df, "PAY_")
    result_pay_amt = wide_to_long(df, "PAY_AMT")

    # Assert
    # result_pay should only contain columns for PAY_1, not PAY_AMT1
    assert list(result_pay.columns) == ["ID", "month", "PAY_"]
    assert len(result_pay) == 1
    assert result_pay.iloc[0]["PAY_"] == 2

    # result_pay_amt should contain columns for PAY_AMT1
    assert list(result_pay_amt.columns) == ["ID", "month", "PAY_AMT"]
    assert len(result_pay_amt) == 1
    assert result_pay_amt.iloc[0]["PAY_AMT"] == 100


def test_wide_to_long_empty_input() -> None:
    # Arrange
    # Missing ID column
    df = pd.DataFrame({"PAY_1": [2]})

    # Act & Assert
    with pytest.raises(KeyError):
        wide_to_long(df, "PAY_")
