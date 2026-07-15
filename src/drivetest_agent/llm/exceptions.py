"""LLM-related error types."""


class LLMServiceError(Exception):
    """Raised when the remote LLM API is unavailable or returns an error."""


class LLMResponseError(LLMServiceError):
    """Raised when an LLM service response has an invalid transport structure."""


class LLMFormatError(Exception):
    """Raised when model output cannot be parsed into a valid TestPlan after retries."""
