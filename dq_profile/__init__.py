"""Isolated SDXL delta-quantization dataset profiler.

The package deliberately lives outside the normal training import path.  The
public training entry points never import it; only ``sdxl_dq_dataset_profile``
does.
"""

SCHEMA_VERSION = "2.1.0"
METRIC_DEFINITION_VERSION = "2.1.0"
PROTOCOL_VERSION = "sdxl-dq-profile-v2.1"

__all__ = ["SCHEMA_VERSION", "METRIC_DEFINITION_VERSION", "PROTOCOL_VERSION"]
