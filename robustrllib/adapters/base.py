"""Dynamics adapter interface."""
from __future__ import annotations

from abc import ABC, abstractmethod


class DynamicsAdapter(ABC):
    """Read/modify a single environment's dynamics parameters by logical name.

    Adapters cache each parameter's *nominal* value on first touch so shifts can
    be expressed relative to nominal (``factor``) and reset exactly.
    """

    @abstractmethod
    def list_params(self) -> tuple[str, ...]:
        """Logical parameter names this adapter understands."""

    @abstractmethod
    def get_nominal(self, param: str, index=None):
        """Nominal (unperturbed) value of ``param``."""

    @abstractmethod
    def set_param(self, param: str, value, index=None) -> None:
        """Write ``param`` into the live model."""

    def read(self, param: str, index=None):
        """CURRENT value of ``param`` in the live model.

        Not the cached nominal: envs may rewrite a field on every reset (Adroit
        Door randomises body_pos[frame]), and a translate shift must add its
        offset to that per-episode value. Adapters that cannot read back raise.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no read(); translate shifts need the live "
            f"value of {param!r}. Implement read() on this adapter.")

    def apply(self, param: str, *, factor=None, value=None, index=None) -> None:
        """Set ``param`` either by scaling its nominal value or to an absolute value."""
        if value is not None:
            self.set_param(param, value, index=index)
        elif factor is not None:
            nominal = self.get_nominal(param, index=index)
            self.set_param(param, nominal * factor, index=index)

    def reset_all(self) -> None:
        """Restore every touched parameter to nominal (no-op by default)."""
