import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Union
import zipfile
import tarfile

from PIL import Image

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from library import train_util


class ArchiveImageLoader:
    def __init__(self, archive_paths: list[str], batch_size: int, preprocess: Callable, debug: bool = False):
        self.archive_paths = archive_paths
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.debug = debug
        self.current_archive = None
        self.archive_index = 0
        self.image_index = 0
        self.files = None
        self.executor = ThreadPoolExecutor()
        self.image_exts = set(train_util.IMAGE_EXTENSIONS)

    def __iter__(self):
        return self

    def __next__(self):
        images = []
        while len(images) < self.batch_size:
            if self.current_archive is None:
                if self.archive_index >= len(self.archive_paths):
                    if len(images) == 0:
                        raise StopIteration
                    else:
                        break  # return the remaining images

                if self.debug:
                    logger.info(f"loading archive: {self.archive_paths[self.archive_index]}")

                current_archive_path = self.archive_paths[self.archive_index]
                if current_archive_path.endswith(".zip"):
                    self.current_archive = zipfile.ZipFile(current_archive_path)
                    self.files = self.current_archive.namelist()
                elif current_archive_path.endswith(".tar"):
                    self.current_archive = tarfile.open(current_archive_path, "r")
                    self.files = self.current_archive.getnames()
                else:
                    raise ValueError(f"unsupported archive file: {self.current_archive_path}")

                self.image_index = 0

                # filter by image extensions
                self.files = [file for file in self.files if os.path.splitext(file)[1].lower() in self.image_exts]

                if self.debug:
                    logger.info(f"found {len(self.files)} images in the archive")

            new_images = []
            while len(images) + len(new_images) < self.batch_size:
                if self.image_index >= len(self.files):
                    break

                file = self.files[self.image_index]
                archive_and_image_path = f"{self.archive_paths[self.archive_index]}////{file}"
                self.image_index += 1

                def load_image(file, archive: Union[zipfile.ZipFile, tarfile.TarFile]):
                    with archive.open(file) as f:
                        image = Image.open(f).convert("RGB")
                        size = image.size
                        image = self.preprocess(image)
                        return image, size

                new_images.append((archive_and_image_path, self.executor.submit(load_image, file, self.current_archive)))

            # wait for all new_images to load to close the archive
            new_images = [(image_path, future.result()) for image_path, future in new_images]

            if self.image_index >= len(self.files):
                self.current_archive.close()
                self.current_archive = None
                self.archive_index += 1

            images.extend(new_images)

        return [(image_path, image, size) for image_path, (image, size) in images]


class ImageLoader:
    def __init__(self, image_paths: list[str], batch_size: int, preprocess: Callable, debug: bool = False):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.debug = debug
        self.image_index = 0
        self.executor = ThreadPoolExecutor()

    def __len__(self):
        return math.ceil(len(self.image_paths) / self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.image_index >= len(self.image_paths):
            raise StopIteration

        images = []
        while len(images) < self.batch_size and self.image_index < len(self.image_paths):

            def load_image(file):
                image = Image.open(file).convert("RGB")
                size = image.size
                image = self.preprocess(image)
                return image, size

            image_path = self.image_paths[self.image_index]
            images.append((image_path, self.executor.submit(load_image, image_path)))
            self.image_index += 1

        images = [(image_path, future.result()) for image_path, future in images]
        return [(image_path, image, size) for image_path, (image, size) in images]


def load_metadata(metadata_file: str):
    if os.path.exists(metadata_file):
        logger.info(f"loading metadata file: {metadata_file}")
        with open(metadata_file, "rt", encoding="utf-8") as f:
            metadata = json.load(f)

        # version check
        major, minor, patch = metadata.get("format_version", "0.0.0").split(".")
        major, minor, patch = int(major), int(minor), int(patch)
        if major > 1 or (major == 1 and minor > 0):
            logger.warning(
                f"metadata format version {major}.{minor}.{patch} is higher than supported version 1.0.0. Some features may not work."
            )

        if "images" not in metadata:
            metadata["images"] = {}
    else:
        logger.info(f"metadata file not found: {metadata_file}, creating new metadata")
        metadata = {"format_version": "1.0.0", "images": {}}

    return metadata


def add_archive_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="metadata file for the dataset. write tags to this file instead of the caption file / データセットのメタデータファイル。キャプションファイルの代わりにこのファイルにタグを書き込む",
    )
    parser.add_argument(
        "--load_archive",
        action="store_true",
        help="load archive file such as .zip instead of image files. currently .zip and .tar are supported. must be used with --metadata"
        " / 画像ファイルではなく.zipなどのアーカイブファイルを読み込む。現在.zipと.tarをサポート。--metadataと一緒に使う必要があります",
    )
