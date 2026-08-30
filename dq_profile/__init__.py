"""Isolated SDXL delta-quantization dataset profiler.

The package deliberately lives outside the normal training import path.  The
public training entry points never import it; only ``sdxl_dq_dataset_profile``
does.
"""

RUNTIME_SCHEMA_VERSION = "2.1.0"
RUNTIME_METRIC_DEFINITION_VERSION = "2.1.0"
PREFIX_GATE_METRIC_VERSION = RUNTIME_METRIC_DEFINITION_VERSION
RUNTIME_PROTOCOL_VERSION = "sdxl-dq-profile-v2.1"

# Backward-compatible aliases for v2.1 artifacts and external readers. New
# code should use the role-specific names above; changing these values would be
# a persisted schema migration rather than a package-version bump.
SCHEMA_VERSION = RUNTIME_SCHEMA_VERSION
METRIC_DEFINITION_VERSION = RUNTIME_METRIC_DEFINITION_VERSION
PROTOCOL_VERSION = RUNTIME_PROTOCOL_VERSION

__all__ = [
    "RUNTIME_SCHEMA_VERSION",
    "RUNTIME_METRIC_DEFINITION_VERSION",
    "PREFIX_GATE_METRIC_VERSION",
    "RUNTIME_PROTOCOL_VERSION",
    "SCHEMA_VERSION",
    "METRIC_DEFINITION_VERSION",
    "PROTOCOL_VERSION",
]
