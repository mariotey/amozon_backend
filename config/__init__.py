"""Use config.ml_config or config.api_config in new code.

Only the two names used by the untouched CB notebook are exposed lazily for
compatibility. Importing config.api_config does not load ML configuration.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ml_config import DATA_OUTPUT_DIR, MODEL_ARTEFACT_DIR

__all__ = ["DATA_OUTPUT_DIR", "MODEL_ARTEFACT_DIR"]


def __getattr__(name):
    """
    Resolve supported legacy configuration names lazily or raise AttributeError.

    Args:
    - name(str): Name identifying the requested resource.

    Returns:
    - Path: Requested legacy data or model directory from ML configuration.
    """
    if name in __all__:
        from . import ml_config

        return getattr(ml_config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
