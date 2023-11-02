import argparse
import json
import os
import re
from pathlib import Path
from typing import List
from tqdm import tqdm
from collections import Counter
import library.train_util as train_util

TAGS_EXT = ".txt"
CAPTION_EXT = ".caption"

PATTERN_HAIR_LENGTH = re.compile(r', (long|short|medium) hair, ')
PATTERN_HAIR_CUT = re.compile(r', (bob|hime) cut, ')
PATTERN_HAIR = re.compile(r', ([\w\-]+) hair, ')
PATTERN_WORD = re.compile(r', ([\w\-]+|hair ornament), ')

PATTERNS_REMOVE_IN_MULTI = [
    PATTERN_HAIR_LENGTH,
    PATTERN_HAIR_CUT,
    re.compile(r', [\w\-]+ eyes, '),
    re.compile(r', ([\w\-]+ sleeves|sleeveless), '),
    re.compile(
        r', (ponytail|braid|ahoge|twintails|[\w\-]+ bun|single hair bun|single side bun|two side up|two tails|[\w\-]+ braid|sidelocks), '),
]

CAPTION_REPLACEMENTS = [
    ('anime anime', 'anime'),
    ('young ', ''),
    ('anime girl', 'girl'),
    ('cartoon female', 'girl'),
    ('cartoon lady', 'girl'),
    ('cartoon character', 'girl'),
    ('cartoon woman', 'girl'),
    ('cartoon women', 'girls'),
    ('cartoon girl', 'girl'),
    ('anime female', 'girl'),
    ('anime lady', 'girl'),
    ('anime character', 'girl'),
    ('anime woman', 'girl'),
    ('anime women', 'girls'),
    ('lady', 'girl'),
    ('female', 'girl'),
    ('woman', 'girl'),
    ('women', 'girls'),
    ('people', 'girls'),
    ('person', 'girl'),
    ('a cartoon figure', 'a figure'),
    ('a cartoon image', 'an image'),
    ('a cartoon picture', 'a picture'),
    ('an anime cartoon image', 'an image'),
    ('a cartoon anime drawing', 'a drawing'),
    ('a cartoon drawing', 'a drawing'),
    ('girl girl', 'girl'),
]

def clean_tags(image_key, tags):
  tags = tags.replace('^_^', '^@@@^')
  tags = tags.replace('_', ' ')
  tags = tags.replace('^@@@^', '^_^')

  tokens = tags.split(", rating")
  if len(tokens) == 1:
    pass
  else:
    if len(tokens) > 2:
      print("multiple ratings:")
      print(f"{image_key} {tags}")
    tags = tokens[0]

  tags = ", " + tags.replace(", ", ", , ") + ", "
  
  if 'girls' in tags or 'boys' in tags:
    for pat in PATTERNS_REMOVE_IN_MULTI:
      found = pat.findall(tags)
      if len(found) > 1:
        tags = pat.sub("", tags)

    srch_hair_len = PATTERN_HAIR_LENGTH.search(tags)
    if srch_hair_len:
      org = srch_hair_len.group()
      tags = PATTERN_HAIR_LENGTH.sub(", @@@, ", tags)

    found = PATTERN_HAIR.findall(tags)
    if len(found) > 1:
      tags = PATTERN_HAIR.sub("", tags)

    if srch_hair_len:
      tags = tags.replace(", @@@, ", org)

  found = PATTERN_WORD.findall(tags)
  for word in found:
    if re.search(f", ((\w+) )+{word}, ", tags):
      tags = tags.replace(f", {word}, ", "")

  tags = tags.replace(", , ", ", ")
  assert tags.startswith(", ") and tags.endswith(", ")
  tags = tags[2:-2]
  return tags

def clean_caption(caption):
  for rf, rt in CAPTION_REPLACEMENTS:
    replaced = True
    while replaced:
      bef = caption
      caption = caption.replace(rf, rt)
      replaced = bef != caption
  return caption

def count_files(image_paths, metadata):
    counts = Counter({'_captions': 0, '_tags': 0})

    for image_key in metadata:
        if 'tags' not in metadata[image_key]:
            counts['_tags'] += 1
        if 'caption' not in metadata[image_key]:
            counts['_captions'] += 1

    return counts

def report_counts(counts, total_files):
    for key, value in counts.items():
        if value == total_files:
            print(f"No {key.replace('_', '')} found for any of the {total_files} images")
        elif value == 0:
            print(f"All {total_files} images have {key.replace('_', '')}")
        else:
            print(f"{total_files - value}/{total_files} images have {key.replace('_', '')}")

def merge_metadata(image_paths, metadata, full_path):
    for image_path in tqdm(image_paths):
        tags_path = image_path.with_suffix(TAGS_EXT)
        if not tags_path.exists():
            tags_path = image_path.joinpath(TAGS_EXT)

        caption_path = image_path.with_suffix(CAPTION_EXT)
        if not caption_path.exists():
            caption_path = image_path.joinpath(CAPTION_EXT)

        image_key = str(image_path) if full_path else image_path.stem
        if image_key not in metadata:
            metadata[image_key] = {}

        if tags_path.is_file():
            tags = tags_path.read_text(encoding='utf-8').strip()
            metadata[image_key]['tags'] = tags

        if caption_path.is_file():
            caption = caption_path.read_text(encoding='utf-8').strip()
            metadata[image_key]['caption'] = caption

    counts = count_files(image_paths, metadata)
    report_counts(counts, len(image_paths))

    return metadata

def clean_metadata(metadata):
    image_keys = list(metadata.keys())
    for image_key in tqdm(image_keys):
        tags = metadata[image_key].get('tags')
        if tags is not None:
            org = tags
            tags = clean_tags(image_key, tags)
            metadata[image_key]['tags'] = tags

        caption = metadata[image_key].get('caption')
        if caption is not None:
            org = caption
            caption = clean_caption(caption)
            metadata[image_key]['caption'] = caption
            
    return metadata

def main(args):
    assert not args.recursive or (args.recursive and args.full_path), "--recursive requires --full_path!"

    train_data_dir_path = Path(args.train_data_dir)
    image_paths: List[Path] = train_util.glob_images_pathlib(train_data_dir_path, args.recursive)
    print(f"Found {len(image_paths)} images.")

    if args.in_json is not None:
        print(f"Loading existing metadata: {args.in_json}")
        metadata = json.loads(Path(args.in_json).read_text(encoding='utf-8'))
        print("Metadata for existing images will be overwritten")
    else:
        print("Creating a new metadata file")
        metadata = {}

    print("Merging tags and captions into metadata json.")
    metadata = merge_metadata(image_paths, metadata, args.full_path)

    if args.clean_caption:
        print("Cleaning captions and tags.")
        metadata = clean_metadata(metadata)

    if args.debug:
        print("Debug: image_key, tags, caption")
        for image_key, data in metadata.items():
            print(image_key, data['tags'], data['caption'])

    print(f"Writing metadata: {args.out_json}")
    Path(args.out_json).write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print("Done!")

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="directory for train images")
    parser.add_argument("out_json", type=str, help="metadata file to output")
    parser.add_argument("--in_json", type=str,
                    help="metadata file to input (if omitted and out_json exists, existing out_json is read)")
    parser.add_argument("--full_path", action="store_true",
                    help="use full path as image-key in metadata (supports multiple directories)")
    parser.add_argument("--recursive", action="store_true",
                    help="recursively search for training tags and captions in all child folders of train_data_dir")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--clean_caption", action="store_true", help="clean captions and tags in metadata")
    
    return parser

if __name__ == '__main__':
    parser = setup_parser()

    args = parser.parse_args()

    main(args)
