import pathlib


def is_safetensors(path):
    path = pathlib.Path(path)
    return path.suffix.lower().endswith("safetensors")
