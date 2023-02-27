import pathlib
import random

import torch

from .base_datasets import BaseDataset
from typing import Union, List
from .common import ImageInfo, KohyaDatasetException, IMAGE_EXTENSIONS


class VAEDataset(BaseDataset):

    def load_vae_dir(self, booth_dir: pathlib.Path):
        if not booth_dir.exists() and not booth_dir.is_dir():
            # print(f"ignore file: {dir}")
            return 0, [], []

        tokens = booth_dir.name.split('_')
        try:
            n_repeats = int(tokens[0])
        except ValueError as e:
            print(f"ignore directory without repeats / 繰り返し回数のないディレクトリを無視します: {booth_dir}")
            return 0, [], []

        folder_caption = '_'.join(tokens[1:])

        img_paths: List[pathlib.Path] = []
        for file in booth_dir.iterdir():
            if file.suffix.lower() in IMAGE_EXTENSIONS:
                img_paths.append(file)

        print(f"found directory {n_repeats}_{folder_caption} contains {len(img_paths)} image files")

        return n_repeats, img_paths

    def __init__(self, batch_size, root_train_data_dir,
                 resolution, enable_bucket, min_bucket_reso, max_bucket_reso,
                 bucket_reso_steps, bucket_no_upscale, flip_aug, color_aug, face_crop_aug_range,
                 random_crop, debug_dataset) -> None:
        super().__init__(None, 255, False, False,
                         resolution, flip_aug, color_aug, face_crop_aug_range, random_crop, debug_dataset)

        self.batch_size = batch_size
        self.size = min(self.width, self.height)  # 短いほう
        self.num_train_images = 0

        self.enable_bucket = enable_bucket

        if self.enable_bucket:
            if min(resolution) >= min_bucket_reso:
                raise KohyaDatasetException(
                    "min_bucket_reso must be either equal or lesser than defined resolution\n"
                    "min_bucket_resoは最小解像度より大きくできません。解像度を大きくするかmin_bucket_resoを小さくしてください"
                )
            elif max(resolution) <= max_bucket_reso:
                raise KohyaDatasetException(
                    "max_bucket_reso must be either equal or greater than defined resolution\n"
                    "max_bucket_resoは最大解像度より小さくできません。解像度を小さくするかmin_bucket_resoを大きくしてください"
                )
            self.min_bucket_reso = min_bucket_reso
            self.max_bucket_reso = max_bucket_reso
            self.bucket_reso_steps = bucket_reso_steps
            self.bucket_no_upscale = bucket_no_upscale
        else:
            self.min_bucket_reso = None
            self.max_bucket_reso = None
            self.bucket_reso_steps = None  # この情報は使われない
            self.bucket_no_upscale = False

        print("Preparing train images...")

        # Prepare files
        train_counts = 0
        for concept_dir in root_train_data_dir.iterdir():
            if concept_dir.is_dir():
                repeats, image_paths = self.load_vae_dir(concept_dir)
                train_counts += repeats * len(image_paths)
                for image_pth, image_paths:
                    image_info = ImageInfo(image_pth.name, repeats, "", False, str(image_pth))
                    self.register_image(image_info)
                self.dataset_dirs_info[concept_dir.name] = {"n_repeats": repeats, "img_count": len(image_paths)}


    def __len__(self):
        return self._data_len + self._data_len_add

    def __getitem__(self, index):
        if index == 0:
            self.shuffle_buckets()

        bucket = self.bucket_manager.buckets[self.buckets_indices[index].bucket_index]
        bucket_batch_size = self.buckets_indices[index].bucket_batch_size
        image_index = self.buckets_indices[index].batch_index * bucket_batch_size
        latents_list = []
        images = []

        for image_key in bucket[image_index: image_index + bucket_batch_size]:
            image_info = self.image_data[image_key]

            # image/latentsを処理する
            if image_info.latents is not None:
                latents = (
                    image_info.latents
                    if not self.flip_aug or random.random() < 0.5
                    else image_info.latents_flipped
                )
                image = None
            elif image_info.latents_npz is not None:
                latents = self.load_latents_from_npz(
                    image_info, self.flip_aug and random.random() >= 0.5
                )
                latents = torch.FloatTensor(latents)
                image = None
            else:
                raise KohyaDatasetException("Latents are required to be cached for VAE dataset! "
                                            "Missing Latents.")

            images.append(image)
            latents_list.append(latents)

        example = {}

        if images[0] is not None:
            images = torch.stack(images)
            images = images.to(memory_format=torch.contiguous_format).float()
        else:
            images = None
        example["images"] = images

        example["latents"] = (
            torch.stack(latents_list) if latents_list[0] is not None else None
        )

        if self.debug_dataset:
            example["image_keys"] = bucket[image_index: image_index + self.batch_size]
        return example


class VAE_TRAIN_DATASET(Dataset):
    def __init__(self, data_path, batch_size=1, gradient_accumulation_steps=1, shuffle=True,
                 resolution=(256, 256), min_resolution=(128, 128), max_size=512, min_size=128, divisible=64,
                 bucket_serch_step=1, make_clipping=0., make_clip_num=1) -> None:
        '''
        data_path = ディレクトリパス　か　画像のパスリスト
        '''
        super().__init__()
        # ファイルパスとか
        self.dataset_dir_path = data_path
        self.file_paths = None
        self.data_list = {}
        self.dots = ["png", "jpg"]
        # 学習に関するデータセット設定
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._data_len: int = 0
        self._data_len_add: int = 0
        self.shuffle: bool = shuffle
        self.make_clipping = make_clipping
        if self.make_clipping >= 1.: self.make_clipping = 0.
        self.make_clip_num = make_clip_num
        # 画像サイズに関する変数
        self.resolution = resolution
        self.min_resolution = min_resolution
        self.max_area_size = (self.resolution[0] // divisible) * (self.resolution[1] // divisible)
        self.min_area_size = (self.min_resolution[0] // divisible) * (self.min_resolution[1] // divisible)
        self.max_size = max_size
        self.min_size = min_size
        self.divisible = divisible
        self.bucket_serch_step = bucket_serch_step
        # bucket関連の変数（画像読み込み時）
        self.buckets_lists = []
        self.area_size_list = []
        self.bucket_area_size_resos_list = []
        self.bucket_area_size_ratio_list = []
        self.add_index = []
        # bucket関連の変数（学習時に使うやつ）
        # index -> vsize key
        self.index_to_enable_bucket_list: list[tuple[int, int]] = []
        # vsize key ->  dataset key
        self.enable_bucket_vsize_to_resos_lens: dict[tuple[int, int], int] = {}
        self.enable_bucket_vsize_to_keys_list: dict[tuple[int, int], list] = {}
        # vsize key -> keys_list index (-> dataset)
        self.enable_bucket_vsize_to_keys_indexs: dict[tuple[int, int], list] = {}
        # 画像データセット
        self.data_list: dict[str, IMAGE_DIC] = {}
        #
        self.image_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        # ファイルのパスをとりあえず取得する
        self.get_files_path()
        # 先にデータリストの入れ物を作る
        self.make_datalist()
        # バケットを作成する
        self.make_buckets()
        # 画像を読み込む
        self.load_images()
        # 有効なバケット情報をまとめる
        self.create_enable_buckets()

        # テスト出力
        # for k, v in self.data_list.items():
        #    print(f"{k}: {v.org_size} -> {v.size} er({v.ratio_error})")
        #

    # 読み込むデータリスト作成関連
    def get_files_path(self):
        file_paths = []
        if type(self.dataset_dir_path) == str:
            for root, dirs, files in os.walk(self.dataset_dir_path, followlinks=True):
                # ファイルを格納
                for file in files:
                    for dot in self.dots:
                        if dot in os.path.splitext(file)[-1]:
                            file_paths.append(os.path.join(root, file))
            self.file_paths = file_paths
        else:
            self.file_paths = self.dataset_dir_path
            self.dataset_dir_path = None

    def make_datalist(self):
        for file_path in self.file_paths:
            key = os.path.splitext(file_path)[0]
            if self.dataset_dir_path == None:
                key = os.path.basename(key)
            else:
                key = key[len(self.dataset_dir_path) + 1:]
            img_data = IMAGE_DIC(file_path)
            self.data_list[key] = img_data

    # バケット作成
    def make_buckets(self):
        _max_area = self.max_area_size
        while _max_area >= self.min_area_size:
            resos = set()
            size = int(math.sqrt(_max_area)) * self.divisible
            resos.add((size, size))
            size = self.min_size
            while size <= self.max_size:
                width = size
                height = min(self.max_size, (_max_area // (size // self.divisible)) * self.divisible)
                if height >= self.min_size:
                    resos.add((width, height))
                    resos.add((height, width))
                size += self.divisible
            resos = list(resos)
            resos.sort()

            self.area_size_list.append(_max_area)
            self.bucket_area_size_resos_list.append(resos)
            ratio = [w / h for w, h in resos]
            self.bucket_area_size_ratio_list.append(np.array(ratio))
            _max_area -= 1

        self.area_size_list = np.array(self.area_size_list)

    # 画像読み込み処理
    def load_image(self, img_path):
        image = Image.open(img_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        return np.array(image, np.uint8)

    def load_images(self):
        print("画像読み込み中...")
        append_list = {}
        for key, img_data in tqdm.tqdm(self.data_list.items()):
            image_org = self.load_image(img_data.path)
            if not type(image_org) == np.ndarray: continue
            # 画像サイズを決める
            img_data.org_size = [image_org.shape[1], image_org.shape[0]]
            img_data.size, img_data.ratio, img_data.ratio_error = self.sel_bucket_size(img_data.org_size[0],
                                                                                       img_data.org_size[1])
            # 画像加工処理
            image, img_data.scale = self.resize_image(image_org, img_data.size, img_data.ratio)
            image = self.image_transforms(image)
            img_data.data = image
            ############################################
            # クリッピング画像追加処理
            if self.make_clipping > 0.:
                if img_data.scale <= self.make_clipping:
                    # 脳筋処理方法(処理工数に無駄はあるけど確実　極端なクリッピングを防ぐためスケーリング処理が必要だった)
                    new_img = image_org
                    # クリッピングサイズが極端すぎないか確認する
                    if img_data.scale < 0.33:
                        new_scale = img_data.scale * 2  # おおよそ半解像度位をクリッピングするのが良さそうな気がする
                        resize = []
                        for i in range(2):
                            resize.append(int(new_img.shape[1 - i] * new_scale + .5))
                        new_img = cv2.resize(new_img, resize, interpolation=cv2.INTER_AREA)
                        # print(f"resize: {img_data.scale} -> {new_scale}")
                    # 追加リスト作成処理
                    for _ in range(self.make_clip_num):
                        i = 0
                        while True:
                            new_key = f"{key}+{i}"
                            if (not new_key in self.data_list) and (not new_key in append_list): break
                            i += 1
                        append_list[new_key] = IMAGE_DIC(new_key)
                        append_list[new_key].data = new_img
                        pos = []
                        for i in range(2):
                            pos.append(random.randint(0, append_list[new_key].data.shape[i] - img_data.size[1 - i] - 1))
                        append_list[new_key].data = append_list[new_key].data[pos[0]:pos[0] + img_data.size[1],
                                                    pos[1]:pos[1] + img_data.size[0]]
                        # ここから下は情報を再取得していく
                        append_list[new_key].org_size = [append_list[new_key].data.shape[1],
                                                         append_list[new_key].data.shape[0]]
                        append_list[new_key].size, append_list[new_key].ratio, append_list[
                            new_key].ratio_error = self.sel_bucket_size(append_list[new_key].org_size[0],
                                                                        append_list[new_key].org_size[1])
                        append_list[new_key].data, append_list[new_key].scale = self.resize_image(
                            append_list[new_key].data, append_list[new_key].size, append_list[new_key].ratio)
                        append_list[new_key].data = self.image_transforms(append_list[new_key].data)
        for k, v in append_list.items():
            self.data_list[k] = v

    def resize_image(self, image, _resized_size, ratio):
        img_size = image.shape[0:2]
        resized_size = [_resized_size[1], _resized_size[0]]  # img_sizeからそのまま呼び出すと並びが　h,w なので処理をすっきりするために入れ替えておく
        # 5/1=2.5 10/3=0.3 1/5=0.2 3/10=0.3
        re_retio = _resized_size[0] / _resized_size[1]
        if re_retio >= ratio:
            base_size = 1
        else:
            base_size = 0
        # cv2の方がPILより高品質な拡大縮小ができる
        resize_scale = resized_size[base_size] / img_size[base_size]
        resize = []
        for i in range(2):
            resize.append(int(img_size[1 - i] * resize_scale + .5))
        if img_size[base_size] > resized_size[base_size]:
            image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)  # INTER_AREAでやりたいのでcv2でリサイズ
        elif img_size[base_size] < resized_size[base_size]:
            image = cv2.resize(image, resize, interpolation=cv2.INTER_CUBIC)  # 遅い代わりに高品質らしい
        #
        img_size = image.shape[0:2]
        p = [0, 0]
        for i in range(len(img_size)):
            if img_size[i] > resized_size[i]:
                trim_size = img_size[i] - resized_size[i]
                p[i] = trim_size // 2
        image = image[p[0]:p[0] + resized_size[0], p[1]:p[1] + resized_size[1]]
        assert image.shape[0] == resized_size[0] and image.shape[1] == resized_size[
            1], f"resized error {image.shape} to {resized_size}"
        return image, resize_scale

    def sel_bucket_size(self, img_width, img_height):
        area_size = (img_width // self.divisible) * (img_height // self.divisible)
        img_ratio = img_width / img_height
        area_size_er = self.area_size_list - area_size
        area_size_id = np.abs(area_size_er).argmin()
        area_size_id_list = [area_size_id]
        # 探査範囲のsize id listを作成する
        for i in range(self.bucket_serch_step):
            if area_size_id - i <= 0:
                area_size_id_list.append(area_size_id + i + 1)
            elif area_size_id + i + 1 >= len(self.bucket_area_size_resos_list):
                area_size_id_list.append(area_size_id - i - 1)
            else:
                area_size_id_list.append(area_size_id - i - 1)
                area_size_id_list.append(area_size_id + i + 1)
        min_error = 10000
        min_area_size_id = area_size_id
        for area_size_id in area_size_id_list:
            area_ratio = self.bucket_area_size_ratio_list[area_size_id]
            ratio_errors = area_ratio - img_ratio
            ratio_error = np.abs(ratio_errors).min()
            if min_error > ratio_error:
                min_error = ratio_error
                min_area_size_id = area_size_id
            if min_error == 0.:
                break
        area_size_id = min_area_size_id
        # ここから普通のバケットサイズ取得
        area_resos = self.bucket_area_size_resos_list[area_size_id]
        area_ratio = self.bucket_area_size_ratio_list[area_size_id]
        ratio_errors = area_ratio - img_ratio
        bucket_id = np.abs(ratio_errors).argmin()
        bucket_size = area_resos[bucket_id]

        return bucket_size, img_ratio, np.abs(ratio_errors).min()

    def make_latent(self, vae):
        print("latent作成中...")
        for img_data in tqdm.tqdm(self.data_list.values()):
            image = img_data.data
            image_tensor = image.unsqueeze(0).to(device=vae.device, dtype=vae.dtype)
            try:
                with torch.no_grad():
                    img_data.latent = vae.encode(image_tensor).latent_dist.mode().squeeze(0).to("cpu")
            except:
                print(f"error: {img_data.path} {image_tensor.size()}")

    # 学習時に使うリストに関する関数
    def create_enable_buckets(self):
        for k, v in self.data_list.items():
            if not v.size in self.enable_bucket_vsize_to_keys_list:
                self.enable_bucket_vsize_to_keys_list[v.size] = [k]
            else:
                self.enable_bucket_vsize_to_keys_list[v.size].append(k)
        for k, v in self.enable_bucket_vsize_to_keys_list.items():
            count = len(v)
            self.enable_bucket_vsize_to_resos_lens[k] = count
            self.reset_indexs_list(k)  # enable bucketsを作成する時に初期化しておく
            self._data_len += (count // self.batch_size) + (count % self.batch_size > 0)
            for _ in range((count // self.batch_size) + (count % self.batch_size > 0)):
                self.index_to_enable_bucket_list.append(k)
        # gradient accumulation stepsのための計算
        self._data_len_add = self.gradient_accumulation_steps - (self._data_len % self.gradient_accumulation_steps)

    def reset_indexs_list(self, vsize):
        now_list = [i for i in range(self.enable_bucket_vsize_to_resos_lens[vsize])]
        self.enable_bucket_vsize_to_keys_indexs[vsize] = now_list

        self.shuffle_indexs_list(vsize)  # リセット時についでにシャッフルしたほうが楽

    def shuffle_indexs_list(self, vsize):
        if self.shuffle:
            now_list = self.enable_bucket_vsize_to_keys_indexs[vsize]
            random.shuffle(now_list)
            self.enable_bucket_vsize_to_keys_indexs[vsize] = now_list
        else:
            pass

    def reset_add_indexs_list(self):
        self.add_index = random.sample(range(self._data_len), self._data_len)

    # 各バケット内の要素を取り出すためのkeyをindexとして扱うための関数
    def get_key(self, vsize):
        keys = self.enable_bucket_vsize_to_keys_list[vsize]
        key_index = self.enable_bucket_vsize_to_keys_indexs[vsize].pop(0)  # popで取り出すので取り出した要素は消える
        # listの中を使い切ったら初期化して補充
        if len(self.enable_bucket_vsize_to_keys_indexs[vsize]) == 0:
            self.reset_indexs_list(vsize)
        return keys[key_index]

    def get_index_to_bucket_key(self, index):
        return self.index_to_enable_bucket_list[index]
