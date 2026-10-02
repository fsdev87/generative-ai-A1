"""The one exception type the services raise for errors that the client should see."""


class ApiError(Exception):
    """Returned to the client as JSON {"detail": message} with the given HTTP status code."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
