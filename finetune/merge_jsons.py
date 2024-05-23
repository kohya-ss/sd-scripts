# simple merge of json files

import argparse
import json
import glob
from tqdm import tqdm

def main(args):
    json_files = glob.glob(args.jsons)
    merged = {}
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for key, values in tqdm(data.items()):
                if key not in merged:
                    merged[key] = data[key]
                elif "train_resolution" in data[key]:
                    merged[key].update(data[key])
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(merged, f, separators=(',', ':'), ensure_ascii=False)

if __name__ == "__main__":
    #python finetune/merge_jsons.py --jsons ${JSON_RESULT_PATH}_*.json --out_json ${JSON_RESULT_PATH}.json
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsons", type=str, help="json files to merge / マージするjsonファイル")
    parser.add_argument("--out_json", type=str, help="output json file / 出力jsonファイル")
    args = parser.parse_args()
    main(args)
