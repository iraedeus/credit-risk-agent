import sqlite3
from unittest.mock import MagicMock, patch

from credit_risk_agent.agent.tools import sql_query


def test_sql_query_success() -> None:
    """Проверяем успешный SELECT запрос и корректное форматирование."""
    # 1. Arrange
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn  # Для работы context manager
    mock_cursor = MagicMock()

    # Имитируем структуру колонок (description) и возвращаемые строки (fetchall)
    mock_cursor.description = [("client_id",), ("age",)]
    mock_cursor.fetchmany.return_value = [(1, 24), (2, 35)]
    mock_conn.cursor.return_value = mock_cursor

    # 2. Act
    with patch("sqlite3.connect", return_value=mock_conn):
        result = sql_query("SELECT client_id, age FROM clients")

    # 3. Assert
    assert "client_id, age" in result  # Заголовок на месте
    assert "1, 24" in result  # Первая строка отформатирована
    assert "2, 35" in result  # Вторая строка отформатирована


def test_sql_query_no_results_or_non_select() -> None:
    """Проверяем поведение, когда запрос выполнен успешно, но не возвращает описание колонок."""
    # 1. Arrange
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()

    # Для INSERT запросов description равен None
    mock_cursor.description = None
    mock_conn.cursor.return_value = mock_cursor

    # 2. Act
    with patch("sqlite3.connect", return_value=mock_conn):
        result = sql_query("INSERT INTO clients (client_id) VALUES (999)")

    # 3. Assert
    assert result == "Запрос выполнен успешно"


def test_sql_query_database_error() -> None:
    """Проверяем, что при ошибке синтаксиса база возвращает сообщение об ошибке, а не падает."""
    # 1. Arrange
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()

    # Настраиваем, чтобы execute вызывал ошибку базы данных
    mock_cursor.execute.side_effect = sqlite3.Error("no such table: clients_bad")
    mock_conn.cursor.return_value = mock_cursor

    # 2. Act
    with patch("sqlite3.connect", return_value=mock_conn):
        result = sql_query("SELECT * FROM clients_bad")

    # 3. Assert
    assert "Ошибка SQL: no such table: clients_bad" in result


def test_sql_query_limit() -> None:
    """Проверяем, что запрос возвращает не более 100 строк (ограничение fetchmany(100))."""
    # 1. Arrange
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()

    mock_cursor.description = [("client_id",)]
    # Имитируем возвращение ровно 100 строк
    mock_cursor.fetchmany.return_value = [(i,) for i in range(100)]
    mock_conn.cursor.return_value = mock_cursor

    # 2. Act
    with patch("sqlite3.connect", return_value=mock_conn):
        result = sql_query("SELECT client_id FROM clients")

    # 3. Assert
    lines = result.split("\n")
    assert lines[0] == "client_id"
    assert len(lines) == 101  # Заголовок + 100 строк данных
    mock_cursor.fetchmany.assert_called_once_with(100)
