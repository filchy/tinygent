class TinyBraveError(Exception):
    """Base class for all TinyBrave exceptions."""


class TinyBraveClientError(TinyBraveError):
    """Exception raised for client-side errors."""


class TinyBraveAPIError(TinyBraveError):
    """Exception raised for API-side errors."""
