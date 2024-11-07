#!/bin/bash
#SBATCH --job-name=cache_latents
#SBATCH --output=O-%x.%j
#SBATCH --error=E-%x.%j
#SBATCH --partition=slurm_rtx4090
#SBATCH --cpus-per-gpu=16
#SBATCH --nodes=1                   # number of nodes
#SBATCH --gres=gpu:1              # number of GPUs per node
#SBATCH --time=72:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --qos=gpu_qos

#unset LD_LIBRARY_PATH # if you have problems with CUDA kernels, let python use installed venv/conda environment's kernel instead.
TRAIN_DATA_DIR=/dataset/path/to/data
JSON_RESULT_PATH=/json/to/save/name
#cd sd-scripts # please CD to your installed script directory

TOTAL_SPLIT=8 # numbers to split, this should match with CUDA_DEVICES_NUM
OFFSET=0 # this is for offsetting, ping me if you need multi-node caching
CUDA_DEVICES_NUM=8 # numbers of GPUs
PYTHON=venv/bin/python # set python
MODEL_PATH=model.safetensors
for i in $(seq 0 $((CUDA_DEVICES_NUM-1)))
do
    INDEX=$((i+OFFSET))
    CUDA_VISIBLE_DEVICES=$i PYTHON finetune/prepare_buckets_latents.py --out_json ${JSON_RESULT_PATH}_${INDEX}.json --split_dataset --n_split $TOTAL_SPLIT --current_index $INDEX --model_name_or_path $MODEL_PATH --max_resolution "1024,1024" --max_bucket_reso 4096 --full_path --recursive --train_data_dir $TRAIN_DATA_DIR &
done

wait

# merge jsons
PYTHON finetune/merge_jsons.py --jsons "${JSON_RESULT_PATH}_*.json" --out_json ${JSON_RESULT_PATH}.json
