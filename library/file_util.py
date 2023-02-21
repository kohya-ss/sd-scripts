import json

import toml

def load_user_config(fname: str) -> dict:
  if fname.lower().endswith('.json'):
    config = json.load(fname)
  elif fname.lower().endswith('.toml'):
    config = toml.load(fname)
  else:
    raise ValueError(f'not supported config file format / 対応していない設定ファイルの形式です: {fname}')

  return config
