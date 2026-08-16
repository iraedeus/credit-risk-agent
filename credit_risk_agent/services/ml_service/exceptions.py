"""Custom exception classes for MLServiceClient."""


class MLServiceClientError(Exception):
    """Base exception class for all MLServiceClient errors."""


class MLServiceHTTPError(MLServiceClientError):
    """
    Exception raised when the ML Service returns an HTTP 4xx or 5xx status code.

    Parameters
    ----------
    status_code : int
        HTTP status code returned by the ML Service.
    message : str
        Detailed error message extracted from the HTTP response.
    """

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(f"[{status_code}] {message}")
