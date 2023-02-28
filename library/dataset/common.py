# Common Classes
# Currently only ImageInfo class
import pathlib
from typing import Tuple, Optional
import torch
import dataclasses

IMAGE_EXTENSIONS = set([".png", ".jpg", ".jpeg", ".webp", ".bmp"])


class KohyaException(Exception):
    pass


class KohyaDatasetException(KohyaException):
    pass


@dataclasses.dataclass
class ImageInfo:

    image_key: str
    num_repeats: int
    caption: str
    is_reg: bool
    absolute_path: str

    image_size: Optional[Tuple[int, int]] = (-1, -1)
    resized_size: Optional[Tuple[int, int]] = (-1, -1)
    bucket_reso: Optional[Tuple[int, int]] = (-1, -1)
    latents: Optional[torch.Tensor] = None
    latents_flipped: Optional[torch.Tensor] = None
    latents_npz: Optional[str] = ""
    latents_npz_flipped: Optional[str] = None


def with_stem(path: pathlib.Path, stem: str):
    return path.with_name(stem + path.suffix)