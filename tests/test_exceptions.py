"""Tests for tactical.exceptions custom exception hierarchy.

Verifies inheritance relationships, instantiation, and that all
exceptions propagate messages correctly.
"""

from __future__ import annotations

import pytest

from tactical.exceptions import (
    AdapterError,
    FeatureExtractionError,
    ModelFitError,
    SegmentationError,
    TacticalError,
)

# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------

SUBCLASSES = [
    AdapterError,
    SegmentationError,
    FeatureExtractionError,
    ModelFitError,
]


class TestExceptionHierarchy:
    """All custom exceptions must inherit from TacticalError."""

    @pytest.mark.parametrize("cls", SUBCLASSES)
    def test_inherits_from_tactical_error(self, cls: type) -> None:
        """Each subclass must be a subclass of TacticalError."""
        assert issubclass(cls, TacticalError)

    @pytest.mark.parametrize("cls", SUBCLASSES)
    def test_inherits_from_exception(self, cls: type) -> None:
        """Each subclass must also be a subclass of Exception."""
        assert issubclass(cls, Exception)

    def test_tactical_error_is_exception(self) -> None:
        """TacticalError itself must inherit from Exception."""
        assert issubclass(TacticalError, Exception)

    @pytest.mark.parametrize("cls", SUBCLASSES)
    def test_subclass_is_not_base(self, cls: type) -> None:
        """Subclasses must be distinct from the base class."""
        assert cls is not TacticalError


# ---------------------------------------------------------------------------
# Catch semantics
# ---------------------------------------------------------------------------


class TestCatchSemantics:
    """Catching TacticalError must catch all subclasses."""

    @pytest.mark.parametrize("cls", SUBCLASSES)
    def test_catch_via_base(self, cls: type) -> None:
        """Raising a subclass must be catchable as TacticalError."""
        with pytest.raises(TacticalError):
            raise cls("test message")

    @pytest.mark.parametrize("cls", SUBCLASSES)
    def test_catch_via_own_type(self, cls: type) -> None:
        """Each exception must be catchable by its own type."""
        with pytest.raises(cls):
            raise cls("test message")

    def test_sibling_not_caught(self) -> None:
        """AdapterError must not be caught as ModelFitError."""
        with pytest.raises(AdapterError):
            raise AdapterError("adapter failed")

        with pytest.raises(ModelFitError):
            raise ModelFitError("model failed")

        # Verify cross-type does not match
        with pytest.raises(AdapterError):
            try:
                raise AdapterError("adapter failed")
            except ModelFitError:
                pytest.fail("AdapterError should not be caught as ModelFitError")


# ---------------------------------------------------------------------------
# Message propagation
# ---------------------------------------------------------------------------


class TestMessagePropagation:
    """Exception messages must be preserved through str() and args."""

    @pytest.mark.parametrize("cls", [TacticalError, *SUBCLASSES])
    def test_message_in_str(self, cls: type) -> None:
        """The message passed at construction must appear in str()."""
        msg = f"test error from {cls.__name__}"
        err = cls(msg)
        assert str(err) == msg

    @pytest.mark.parametrize("cls", [TacticalError, *SUBCLASSES])
    def test_message_in_args(self, cls: type) -> None:
        """The message must be accessible via the args tuple."""
        msg = "something went wrong"
        err = cls(msg)
        assert err.args == (msg,)

    @pytest.mark.parametrize("cls", [TacticalError, *SUBCLASSES])
    def test_empty_message(self, cls: type) -> None:
        """Exceptions must be constructable with no message."""
        err = cls()
        assert str(err) == ""
        assert err.args == ()


# ---------------------------------------------------------------------------
# Docstrings
# ---------------------------------------------------------------------------


class TestDocstrings:
    """All exception classes must have docstrings."""

    @pytest.mark.parametrize("cls", [TacticalError, *SUBCLASSES])
    def test_has_docstring(self, cls: type) -> None:
        """Each exception class must define a non-empty docstring."""
        assert cls.__doc__ is not None
        assert len(cls.__doc__.strip()) > 0
