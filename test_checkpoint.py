import sdxl_gen_img
import cv2  
import os
import numpy as np
import shutil
import gc

# 初始化原測試程式
parser = sdxl_gen_img.setup_parser()
args = parser.parse_args()
workspace = "D:/AIGC/model/slot_checkpoints/"

# args.ckpt = "D:/AIGC/model/slot_checkpoints/slot_checkpoints-step00043860.safetensors"


# 覆蓋原測試程式參數
args.xformers = True
args.bf16 = True
args.no_half_vae = True
args.W = 1024
args.H = 1024
args.scale = 7
args.sampler = "ddim"
args.steps = 30
args.batch_size = 2
args.images_per_prompt = 1
args.seed = 0

temp_dir = os.path.join(workspace, "temp")  

# 測試自然語言prompt
args.from_file = "./prompts/NaturalLanguagePrompts.txt"
args.outdir = os.path.join(temp_dir, "NaturalLanguagePrompts")
if os.path.exists(args.outdir):
    shutil.rmtree(args.outdir)
del sdxl_gen_img
import sdxl_gen_img
gc.collect()
sdxl_gen_img.main(args)

# 測試一般prompt
args.from_file = "./prompts/NormalPrompt.txt"
args.outdir = os.path.join(temp_dir, "NormalPrompts")
if os.path.exists(args.outdir):
    shutil.rmtree(args.outdir)
del sdxl_gen_img
import sdxl_gen_img
gc.collect()
sdxl_gen_img.main(args)

# 測試動物prompt
args.from_file = "./prompts/AnimalPrompts.txt"
args.outdir = os.path.join(temp_dir, "AnimalPrompts")
if os.path.exists(args.outdir):
    shutil.rmtree(args.outdir)
del sdxl_gen_img
import sdxl_gen_img
gc.collect()
sdxl_gen_img.main(args)

# 測試風格prompt
args.from_file = "./prompts/StylePrompts.txt"
args.outdir = os.path.join(temp_dir, "StylePrompts")
if os.path.exists(args.outdir):
    shutil.rmtree(args.outdir)
del sdxl_gen_img
import sdxl_gen_img
gc.collect()
sdxl_gen_img.main(args)

# 測試slot風格prompt
args.from_file = "./prompts/SlotStylePrompts.txt"
args.outdir = os.path.join(temp_dir, "SlotStylePrompts")
if os.path.exists(args.outdir):
    shutil.rmtree(args.outdir)
del sdxl_gen_img
import sdxl_gen_img
gc.collect()
sdxl_gen_img.main(args)



#將所有圖片合併成一張 並在一組圖片上標示所測試的任務(資料夾名稱) 每一組圖片一行
def merge_images_row(path, title_size=100): #合併同一資料夾的圖片 path: 資料夾路徑
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        images.append(img)
    max_h = max([img.shape[0] for img in images]) # 取最大高度
    sum_w = sum([img.shape[1] for img in images]) # 總寬度
    row_image = np.zeros((max_h, sum_w, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        h, w, c = img.shape
        row_image[:h, i*w:(i+1)*w] = img # 橫向合併
    

    #新增一行標題
    title = np.zeros((title_size, row_image.shape[1], 3), dtype=np.uint8)
    basename = os.path.basename(path)
    cv2.putText(title, basename, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    row_image = np.concatenate([title, row_image], axis=0) # 縱向合併

    # h, w, c = row_image.shape
    # cv2.imshow('image', cv2.resize(row_image, (1024, (1024 + title_size) // len(images)))) # 顯示圖片(縮小
    # cv2.waitKey(0)
    return row_image

all_images = []
for foldername in os.listdir(temp_dir):
    row_image = merge_images_row(os.path.join(temp_dir, foldername))
    all_images.append(row_image)

max_w = max([img.shape[1] for img in all_images]) # 取最大寬度
sum_h = sum([img.shape[0] for img in all_images]) # 總高度
grid_image = np.zeros((sum_h, max_w, 3), dtype=np.uint8)
for i, img in enumerate(all_images):
    h, w, c = img.shape
    grid_image[i*h:(i+1)*h, :w] = img # 縱向合併

ckpt_basename = os.path.basename(args.ckpt)
ckpt_name = ckpt_basename.split('.')[0]
save_path = os.path.join(workspace, "test", f"{ckpt_name}.png")
cv2.imwrite(save_path, grid_image)


