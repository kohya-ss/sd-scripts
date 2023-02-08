import argparse
import glob
import os
import json
import random

import sys
sys.path.insert(0, os.path.abspath('./ofa/fairseq'))

from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
from torchvision import transforms
from fairseq import utils, tasks
from fairseq import checkpoint_utils
from ofa.utils.eval_utils import eval_step
from ofa.tasks.mm_tasks.caption import CaptionTask
from ofa.data import data_utils

from urllib.parse import urlparse
from timm.models.hub import download_cached_file
from torchvision.transforms.functional import InterpolationMode

import library.train_util as train_util

# Register caption task
tasks.register_task('caption', CaptionTask)
# turn on cuda if GPU is available
use_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_SIZE = 480
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

def is_url(url_or_filename):
  parsed = urlparse(url_or_filename)
  return parsed.scheme in ("http", "https")

# 共通化したいが微妙に処理が異なる……
class ImageLoadingTransformDataset(torch.utils.data.Dataset):
  def __init__(self, image_paths):
    self.images = image_paths

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    img_path = self.images[idx]

    try:
      image = Image.open(img_path).convert("RGB")
      # convert to tensor temporarily so dataloader will accept it
      tensor = IMAGE_TRANSFORM(image)
    except Exception as e:
      print(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
      return None

    return (tensor, img_path)


def collate_fn_remove_corrupted(batch):
  """Collate function that allows to remove corrupted examples in the
  dataloader. It expects that the dataloader returns 'None' when that occurs.
  The 'None's in the batch are removed.
  """
  # Filter out all the Nones (corrupted examples)
  batch = list(filter(lambda x: x is not None, batch))
  return batch

def collate(samples, pad_idx, eos_idx):
  if len(samples) == 0:
    return {}
  def merge(key):
    return data_utils.collate_tokens(
      [s[key] for s in samples],
      pad_idx,
      eos_idx=eos_idx,
  )
  id = np.array([s["id"] for s in samples])
  src_tokens = merge("source")
  src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])
  patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
  patch_masks = torch.cat([sample['patch_mask'] for sample in samples])
  prev_output_tokens = None
  target = None
  if samples[0].get("target", None) is not None:
    target = merge("target")
    tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])
    ntokens = tgt_lengths.sum().item()
    if samples[0].get("prev_output_tokens", None) is not None:
      prev_output_tokens = merge("prev_output_tokens")
  else:
    ntokens = src_lengths.sum().item()
  batch = {
    "id": id,
    "nsentences": len(samples),
    "ntokens": ntokens,
    "net_input": {
      "src_tokens": src_tokens,
      "src_lengths": src_lengths,
      "patch_images": patch_images,
      "patch_masks": patch_masks,
      "prev_output_tokens": prev_output_tokens
    },
    "target": target,
  }
  return batch

# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t

def main(args):
  # fix the seed for reproducibility
  seed = args.seed  # + utils.get_rank()
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  
  use_fp16 = args.fp16

  print(f"load images from {args.train_data_dir}")
  image_paths = []
  # using set will speed up the filtering process
  caption_paths = set(glob.glob(os.path.join(args.train_data_dir, "*" + args.caption_extension)))
  for ip in train_util.glob_images(args.train_data_dir):
    caption_path = "".join([os.path.splitext(ip)[0], ".caption"])
    if caption_path not in caption_paths:
      image_paths.append(ip)
  print(f"Need to process {len(image_paths)} images.")

  print(f"loading OFA caption: {args.caption_weights}")
  if is_url(args.caption_weights):
    args.caption_weights = download_cached_file(args.caption_weights, check_hash=False, progress=True)
  overrides = {"bpe_dir": "ofa/utils/BPE", "eval_cider": False, "beam": args.num_beams,
                "max_len_b": args.max_length, "no_repeat_ngram_size": args.no_repeat_ngram_size, "seed": args.seed, "temperature":args.temperature, "min_len": args.min_length}
  models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
      utils.split_paths(args.caption_weights),
      arg_overrides=overrides
  )
  generator = task.build_generator(models, cfg.generation)
  for model in models:
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)
  print("OFA loaded")
  
  # Text preprocess
  bos_item = torch.LongTensor([task.src_dict.bos()])
  eos_item = torch.LongTensor([task.src_dict.eos()])
  pad_idx = task.src_dict.pad()
  
  def encode_text(text, length=None, append_bos=False, append_eos=False):
    s = task.tgt_dict.encode_line(
        line=task.bpe.encode(text),
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s
  
  def construct_sample(id, image: Image):
    patch_mask = torch.tensor([True])
    src_item = torch.cat([bos_item, encode_text("what does the image describe?"), eos_item])
    # src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id": id,
        "source": src_item,
        "patch_image": image,
        "patch_mask": patch_mask
    }
    return sample

  # captioningする
  def run_batch(path_imgs):
    imgs = [construct_sample(str(i),im[1]) for i,im in enumerate(path_imgs)]
    sample = collate(imgs, pad_idx,task.src_dict.eos())
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

    with torch.no_grad():
      captions , _ = eval_step(task, generator, models, sample)
    idx = 0
    for image_path, _ in path_imgs:
      caption = captions[idx]['caption']
      with open(os.path.splitext(image_path)[0] + args.caption_extension, "wt", encoding='utf-8') as f:
        f.write(caption + "\n")
        if args.debug:
          print(image_path, caption)
      idx += args.num_beams

  # 読み込みの高速化のためにDataLoaderを使うオプション
  if args.max_data_loader_n_workers is not None:
    dataset = ImageLoadingTransformDataset(image_paths)
    data = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.max_data_loader_n_workers, collate_fn=collate_fn_remove_corrupted, drop_last=False)
  else:
    data = [[(None, ip)] for ip in image_paths]

  b_imgs = []
  for data_entry in tqdm(data, smoothing=0.0):
    for data in data_entry:
      if data is None:
        continue

      img_tensor, image_path = data
      if img_tensor is None:
        try:
          raw_image = Image.open(image_path)
          if raw_image.mode != 'RGB':
            raw_image = raw_image.convert("RGB")
          img_tensor = IMAGE_TRANSFORM(raw_image)
        except Exception as e:
          print(f"Could not load image path / 画像を読み込めません: {image_path}, error: {e}")
          continue

      b_imgs.append((image_path, img_tensor))
      if len(b_imgs) >= args.batch_size:
        run_batch(b_imgs)
        b_imgs.clear()
  if len(b_imgs) > 0:
    run_batch(b_imgs)

  print("done!")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
  parser.add_argument("--caption_weights", type=str, default="https://huggingface.co/sheldonxxxx/ofa_for_repo/resolve/main/caption_huge_best.pt",
                      help="OFA caption weights (caption_huge_best.pth) / OFA captionの重みファイル(model_large_caption.pth)")
  parser.add_argument("--caption_extention", type=str, default=None,
                      help="extension of caption file (for backward compatibility) / 出力されるキャプションファイルの拡張子（スペルミスしていたのを残してあります）")
  parser.add_argument("--caption_extension", type=str, default=".caption", help="extension of caption file / 出力されるキャプションファイルの拡張子")
  parser.add_argument("--batch_size", type=int, default=3, help="batch size in inference / 推論時のバッチサイズ")
  parser.add_argument("--max_data_loader_n_workers", type=int, default=None,
                      help="enable image reading by DataLoader with this number of workers (faster) / DataLoaderによる画像読み込みを有効にしてこのワーカー数を適用する（読み込みを高速化）")
  parser.add_argument("--num_beams", type=int, default=5, help="num of beams in beam search /beam search時のビーム数（多いと精度が上がるが時間がかかる）")
  parser.add_argument("--temperature", type=float, default=0.5, help="top_p in Nucleus sampling / Nucleus sampling時のtop_p")
  parser.add_argument("--max_length", type=int, default=75, help="max length of caption / captionの最大長")
  parser.add_argument("--min_length", type=int, default=5, help="min length of caption / captionの最小長")
  parser.add_argument('--seed', default=42, type=int, help='seed for reproducibility / 再現性を確保するための乱数seed')
  parser.add_argument('--no_repeat_ngram_size', default=3, type=int, help='')
  parser.add_argument("--fp16", action="store_true", help="inference with fp16")
  parser.add_argument("--debug", action="store_true", help="debug mode")

  args = parser.parse_args()

  # スペルミスしていたオプションを復元する
  if args.caption_extention is not None:
    args.caption_extension = args.caption_extention

  main(args)
