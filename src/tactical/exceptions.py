"""Custom exceptions for the Tactical State Discovery Engine.

All exceptions inherit from :class:`TacticalError` so callers can catch
the full family with a single ``except TacticalError`` clause.
"""


class TacticalError(Exception):
    """Base exception for all tactical engine errors."""


class AdapterError(TacticalError):
    """Raised when a data adapter fails to load or transform data."""


class SegmentationError(TacticalError):
    """Raised when match segmentation encounters an unrecoverable error."""


class FeatureExtractionError(TacticalError):
    """Raised when feature extraction fails for a segment or match."""


class ModelFitError(TacticalError):
    """Raised when a model cannot be fitted to the provided data."""
