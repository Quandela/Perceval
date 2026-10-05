"""Backward-compatible remote-processor exports.

``RemoteProcessor`` was replaced by the remote-computer API.  Keep the
performance metadata key importable for integrations that still target the
former module path.
"""

from .communication_layer import PERFS_KEY

__all__ = ["PERFS_KEY"]
