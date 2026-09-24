"""
OpenCV (``cv2``) optional-dependency bridge.

Importing this module has one of two effects:

1. If ``opencv-python`` is installed, nothing changes -- the real ``cv2``
   module is used as-is, and :data:`HAS_OPENCV` is set to ``True``.

2. If it is not installed, a lightweight stub implemented in
   :mod:`library._cv2_stub` is registered as ``cv2`` in
   :data:`sys.modules`, so any subsequent ``import cv2`` call transparently
   picks it up. :data:`HAS_OPENCV` is set to ``False``.

Import this module BEFORE ``import cv2`` in any file that wants to tolerate
a missing OpenCV installation::

    from library import cv2_compat  # noqa: F401 - must come before `import cv2`
    import cv2

Tools that genuinely require OpenCV should bail out early with a clear
message when :data:`HAS_OPENCV` is ``False``.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

try:
    import cv2 as _real_cv2  # noqa: F401  # real OpenCV is importable
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    if "cv2" not in sys.modules:
        from library import _cv2_stub

        sys.modules["cv2"] = _cv2_stub
        logger.info(
            "opencv-python is not installed; using the built-in cv2 stub "
            "(Pillow/NumPy fallback). Some tools that rely on the real "
            "OpenCV will not work."
        )


def require_opencv(tool_name: str = "this tool") -> None:
    """Abort execution with a clear message when real OpenCV is missing."""
    if HAS_OPENCV:
        return
    raise SystemExit(
        f"{tool_name} requires opencv-python, but it is not installed.\n"
        f"{tool_name} は opencv-python が必要ですが、インストールされていません。\n"
        "Install it with: pip install opencv-python"
    )


__all__ = ["HAS_OPENCV", "require_opencv"]
