"""Custom exception classes for DataServiceClient."""


class DataServiceClientError(Exception):
    """Base exception class for all DataServiceClient errors."""


class DataServiceHTTPError(DataServiceClientError):
    """
    Exception raised when Data Service returns an HTTP 4xx or 5xx status code.

    Parameters
    ----------
    status_code : int
        HTTP status code returned by the Data Service.
    message : str
        Detailed error message extracted from the HTTP response.
    """

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(f"[{status_code}] {message}")
