import json
from pathlib import Path

import pandas as pd
import pytest

from data.normalization import StandardScaler, fit_and_save_scaler, normalize


def test_fit_and_save_scaler(tmp_path: Path) -> None:
    # Arrange
    df = pd.DataFrame({"feature1": [10.0, 20.0, 30.0], "feature2": [1.0, 2.0, 3.0]})
    columns = ["feature1", "feature2"]
    save_path = tmp_path / "scaler.json"

    # Act
    fit_and_save_scaler(df, columns, save_path)

    # Assert
    assert save_path.exists()
    with save_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert "mean" in data
    assert "std" in data
    assert data["mean"]["feature1"] == pytest.approx(20.0)
    assert data["mean"]["feature2"] == pytest.approx(2.0)
    assert data["std"]["feature1"] == pytest.approx(10.0)
    assert data["std"]["feature2"] == pytest.approx(1.0)


def test_normalize_core(tmp_path: Path) -> None:
    # Arrange
    df = pd.DataFrame({"feature1": [10.0, 20.0, 30.0], "other": [100, 200, 300]})
    columns = ["feature1"]

    # Сохраняем мок-файл параметров
    scaler_path = tmp_path / "scaler.json"
    scaler_data = {"mean": {"feature1": 20.0}, "std": {"feature1": 10.0}}
    with scaler_path.open("w", encoding="utf-8") as f:
        json.dump(scaler_data, f)

    # Act
    result = normalize(df, columns, scaler_path)

    # Assert
    # Проверяем корректность нормализации целевой колонки
    assert result["feature1"].tolist() == pytest.approx([-1.0, 0.0, 1.0])
    # Проверяем, что нецелевая колонка осталась нетронутой
    assert result["other"].tolist() == [100, 200, 300]


def test_normalize_zero_std_robustness(tmp_path: Path) -> None:
    # Arrange
    # feature1 имеет нулевое стандартное отклонение
    df = pd.DataFrame({"feature1": [5.0, 5.0, 5.0]})
    columns = ["feature1"]

    # fit_and_save_scaler должен защищать от деления на 0
    scaler_path = tmp_path / "scaler.json"
    fit_and_save_scaler(df, columns, scaler_path)

    # Act
    result = normalize(df, columns, scaler_path)

    # Assert
    # Деление на 0 не должно приводить к NaN/inf
    assert result["feature1"].tolist() == [0.0, 0.0, 0.0]


def test_normalize_non_inplace_safety(tmp_path: Path) -> None:
    # Arrange
    df = pd.DataFrame({"feature1": [10.0, 20.0, 30.0]})
    columns = ["feature1"]
    sliced_df = df.iloc[0:2]

    scaler_path = tmp_path / "scaler.json"
    scaler_data = {"mean": {"feature1": 20.0}, "std": {"feature1": 10.0}}
    with scaler_path.open("w", encoding="utf-8") as f:
        json.dump(scaler_data, f)

    # Act
    result = normalize(sliced_df, columns, scaler_path)

    # Assert
    # Исходный срез не должен измениться (защита от SettingWithCopyWarning)
    assert sliced_df["feature1"].tolist() == [10.0, 20.0]
    # Возвращенный DataFrame нормализован корректно
    assert result["feature1"].tolist() == pytest.approx([-1.0, 0.0])


def test_normalize_missing_column_error(tmp_path: Path) -> None:
    # Arrange
    df = pd.DataFrame({"feature1": [10.0, 20.0], "missing_feature": [1.0, 2.0]})
    columns = ["feature1", "missing_feature"]

    scaler_path = tmp_path / "scaler.json"
    scaler_data = {"mean": {"feature1": 20.0}, "std": {"feature1": 10.0}}
    with scaler_path.open("w", encoding="utf-8") as f:
        json.dump(scaler_data, f)

    # Act & Assert
    with pytest.raises(ValueError, match="absent in the scaler parameters"):
        normalize(df, columns, scaler_path)


def test_standard_scaler_class(tmp_path: Path) -> None:
    # Arrange
    df = pd.DataFrame({"feature1": [10.0, 20.0, 30.0]})
    columns = ["feature1"]
    save_path = tmp_path / "scaler_class.json"

    # Act - Fit & Save
    scaler = StandardScaler().fit(df, columns)
    scaler.save(save_path)

    # Assert save
    assert save_path.exists()

    # Act - Load & Transform
    loaded_scaler = StandardScaler.load(save_path)
    result = loaded_scaler.transform(df, columns)

    # Assert transform
    assert result["feature1"].tolist() == pytest.approx([-1.0, 0.0, 1.0])
