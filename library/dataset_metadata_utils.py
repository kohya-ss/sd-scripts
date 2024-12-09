import os
import json
from typing import Any, Optional


from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


METADATA_VERSION = [1, 0, 0]
VERSION_STRING = ".".join(str(v) for v in METADATA_VERSION)

ARCHIVE_PATH_SEPARATOR = "////"


def load_metadata(metadata_file: str, create_new: bool = False) -> Optional[dict[str, Any]]:
    if os.path.exists(metadata_file):
        logger.info(f"loading metadata file: {metadata_file}")
        with open(metadata_file, "rt", encoding="utf-8") as f:
            metadata = json.load(f)

        # version check
        major, minor, patch = metadata.get("format_version", "0.0.0").split(".")
        major, minor, patch = int(major), int(minor), int(patch)
        if major > METADATA_VERSION[0] or (major == METADATA_VERSION[0] and minor > METADATA_VERSION[1]):
            logger.warning(
                f"metadata format version {major}.{minor}.{patch} is higher than supported version {VERSION_STRING}. Some features may not work."
            )

        if "images" not in metadata:
            metadata["images"] = {}
    else:
        if not create_new:
            return None
        logger.info(f"metadata file not found: {metadata_file}, creating new metadata")
        metadata = {"format_version": VERSION_STRING, "images": {}}

    return metadata


def is_archive_path(archive_and_image_path: str) -> bool:
    return archive_and_image_path.count(ARCHIVE_PATH_SEPARATOR) == 1


def get_inner_path(archive_and_image_path: str) -> str:
    return archive_and_image_path.split(ARCHIVE_PATH_SEPARATOR, 1)[1]


def get_archive_digest(archive_and_image_path: str) -> str:
    """
    calculate a 8-digits hex digest for the archive path to avoid collisions for different archives with the same name.
    """
    archive_path = archive_and_image_path.split(ARCHIVE_PATH_SEPARATOR, 1)[0]
    return f"{hash(archive_path) & 0xFFFFFFFF:08x}"
