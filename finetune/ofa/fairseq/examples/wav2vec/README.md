# wav2vec 2.0

wav2vec 2.0 learns speech representations on unlabeled data as described in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)](https://arxiv.org/abs/2006.11477).

We learned speech representations in multiple languages as well in [Unsupervised Cross-lingual Representation Learning for Speech Recognition (Conneau et al., 2020)](https://arxiv.org/abs/2006.13979).

We also combined wav2vec 2.0 with self-training in [Self-training and Pre-training are Complementary for Speech Recognition (Xu et al., 2020)](https://arxiv.org/abs/2010.11430).

We combined speech data from multiple domains in [Robust wav2vec 2.0: Analyzing Domain Shift in Self-Supervised Pre-Training (Hsu, et al., 2021)](https://arxiv.org/abs/2104.01027)

## Pre-trained models

Model | Finetuning split | Dataset | Model
|---|---|---|---
Wav2Vec 2.0 Base | No finetuning | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt)
Wav2Vec 2.0 Base | 10 minutes | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_10m.pt)
Wav2Vec 2.0 Base | 100 hours | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_100h.pt)
Wav2Vec 2.0 Base | 960 hours | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_960h.pt)
Wav2Vec 2.0 Large | No finetuning | [Librispeech](http://www.openslr.org/12)  | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt)
Wav2Vec 2.0 Large | 10 minutes | [Librispeech](http://www.openslr.org/12)  | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_big_10m.pt)
Wav2Vec 2.0 Large | 100 hours | [Librispeech](http://www.openslr.org/12)  | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_big_100h.pt)
Wav2Vec 2.0 Large | 960 hours | [Librispeech](http://www.openslr.org/12)  | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_big_960h.pt)
Wav2Vec 2.0 Large (LV-60)* | No finetuning | [Libri-Light](https://github.com/facebookresearch/libri-light) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt)
Wav2Vec 2.0 Large (LV-60)* | 10 minutes | [Libri-Light](https://github.com/facebookresearch/libri-light) + [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_10m_new.pt)
Wav2Vec 2.0 Large (LV-60)* | 100 hours | [Libri-Light](https://github.com/facebookresearch/libri-light) + [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_100h_new.pt)
Wav2Vec 2.0 Large (LV-60)* | 960 hours | [Libri-Light](https://github.com/facebookresearch/libri-light) + [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec2_vox_960h_new.pt)
Wav2Vec 2.0 Large (LV-60) + Self Training * | 10 minutes | [Libri-Light](https://github.com/facebookresearch/libri-light) + [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_10m_pl.pt)
Wav2Vec 2.0 Large (LV-60) + Self Training * | 100 hours | [Libri-Light](https://github.com/facebookresearch/libri-light) + [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_100h_pl.pt)
Wav2Vec 2.0 Large (LV-60) + Self Training * | 960 hours | [Libri-Light](https://github.com/facebookresearch/libri-light) + [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_960h_pl.pt)
Wav2Vec 2.0 Large (LV-60 + CV + SWBD + FSH) ** | No finetuning | [Libri-Light](https://github.com/facebookresearch/libri-light) + [CommonVoice](https://commonvoice.mozilla.org/en/languages) + [Switchboard](https://catalog.ldc.upenn.edu/LDC97S62) + [Fisher](https://catalog.ldc.upenn.edu/LDC2004T19) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/w2v_large_lv_fsh_swbd_cv.pt)
Wav2Vec 2.0 Large (LV-60 + CV + SWBD + FSH) ** | 960 hours Librispeech | [Libri-Light](https://github.com/facebookresearch/libri-light) + [CommonVoice](https://commonvoice.mozilla.org/en/languages) + [Switchboard](https://catalog.ldc.upenn.edu/LDC97S62) + [Fisher](https://catalog.ldc.upenn.edu/LDC2004T19) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/w2v_large_lv_fsh_swbd_cv_ftls960.pt)
Wav2Vec 2.0 Large (LV-60 + CV + SWBD + FSH) ** | 300 hours Switchboard | [Libri-Light](https://github.com/facebookresearch/libri-light) + [CommonVoice](https://commonvoice.mozilla.org/en/languages) + [Switchboard](https://catalog.ldc.upenn.edu/LDC97S62) + [Fisher](https://catalog.ldc.upenn.edu/LDC2004T19) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/w2v_large_lv_fsh_swbd_cv_ftsb300.pt)

\* updated (Oct. 24, 2020)\
** updated (Jul. 8, 2021)

We also release multilingual pre-trained wav2vec 2.0 (XLSR) models:

Model | Architecture | Hours | Languages | Datasets | Model
|---|---|---|---|---|---
XLSR-53 | Large | 56k | 53 | MLS, CommonVoice, BABEL | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt)

The XLSR model uses the following datasets for multilingual pretraining:

* **[MLS: Multilingual LibriSpeech](https://indico2.conference4me.psnc.pl/event/35/contributions/3585/attachments/1060/1101/Wed-2-6-10.pdf)** (8 languages, 50.7k hours): *Dutch, English, French, German, Italian, Polish, Portuguese, Spanish*

* **[CommonVoice](https://commonvoice.mozilla.org/en/languages)** (36 languages, 3.6k hours): *Arabic, Basque, Breton, Chinese (CN), Chinese (HK), Chinese (TW), Chuvash, Dhivehi, Dutch, English, Esperanto, Estonian, French, German, Hakh-Chin, Indonesian, Interlingua, Irish, Italian, Japanese, Kabyle, Kinyarwanda, Kyrgyz, Latvian, Mongolian, Persian, Portuguese, Russian, Sakha, Slovenian, Spanish, Swedish, Tamil, Tatar, Turkish, Welsh* (see also [finetuning splits]([https://dl.fbaipublicfiles.com/cpc_audio/common_voices_splits.tar.gz]) from [this paper](https://arxiv.org/abs/2002.02848)).

* **[Babel](https://catalog.ldc.upenn.edu/byyear)** (17 languages, 1.7k hours): *Assamese, Bengali, Cantonese, Cebuano, Georgian, Haitian, Kazakh, Kurmanji, Lao, Pashto, Swahili, Tagalog, Tamil, Tok, Turkish, Vietnamese, Zulu*


## Training a new model with the CLI tools

Given a directory containing wav files to be used for pretraining (we recommend splitting each file into separate file 10 to 30 seconds in length)

### Prepare training data manifest:

First, install the `soundfile` library:
```shell script
pip install soundfile
```

Next, run:

```shell script
$ python examples/wav2vec/wav2vec_manifest.py /path/to/waves --dest /manifest/path --ext $ext --valid-percent $valid
```

$ext should be set to flac, wav, or whatever format your dataset happens to use that soundfile can read.

$valid should be set to some reasonable percentage (like 0.01) of training data to use for validation.
To use a pre-defined validation set (like dev-other from librispeech), set to it 0 and then overwrite valid.tsv with a
separately pre-processed manifest file.

### Train a wav2vec 2.0 base model:

This configuration was used for the base model trained on the Librispeech dataset in the wav2vec 2.0 paper

Note that the input is expected to be single channel, sampled at 16 kHz

```shell script
$ fairseq-hydra-train \
    task.data=/path/to/data \
    --config-dir /path/to/fairseq-py/examples/wav2vec/config/pretraining \
    --config-name wav2vec2_base_librispeech
```

Note: you can simulate 64 GPUs by using k GPUs and adding command line parameters (before `--config-dir`)
`distributed_training.distributed_world_size=k` `+optimization.update_freq='[x]'` where x = 64/k

### Train a wav2vec 2.0 large model:

This configuration was used for the large model trained on the Libri-light dataset in the wav2vec 2.0 paper

```shell script
$ fairseq-hydra-train \
    task.data=/path/to/data \
    --config-dir /path/to/fairseq-py/examples/wav2vec/config/pretraining \
    --config-name wav2vec2_large_librivox
```

Note: you can simulate 128 GPUs by using k GPUs and adding command line parameters (before `--config-dir`)
`distributed_training.distributed_world_size=k` `+optimization.update_freq='[x]'` where x = 128/k

### Fine-tune a pre-trained model with CTC:

Fine-tuning a model requires parallel audio and labels file, as well as a vocabulary file in fairseq format.
A letter vocabulary can be downloaded [here](https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt).
An example [script](libri_labels.py) that generates labels for the Librispeech dataset from the tsv file produced by wav2vec_manifest.py can be used as follows:

```shell script
split=train
$ python libri_labels.py /path/to/tsv --output-dir /output/dir --output-name $split
```

Fine-tuning on 100h of Librispeech with letter targets:
```shell script
$ fairseq-hydra-train \
    distributed_training.distributed_port=$PORT \
    task.data=/path/to/data \
    model.w2v_path=/path/to/model.pt \
    --config-dir /path/to/fairseq-py/examples/wav2vec/config/finetuning \
    --config-name base_100h
```

There are other config files in the config/finetuning directory that can be used to fine-tune on other splits.
You can specify the right config via the `--config-name` parameter.

Note: you can simulate 24 GPUs by using k GPUs and adding command line parameters (before `--config-dir`)
`distributed_training.distributed_world_size=k` `+optimization.update_freq='[x]'` where x = 24/k

Decoding with a language model during training requires flashlight [python bindings](https://github.com/facebookresearch/flashlight/tree/master/bindings/python) (previously called [wav2letter](https://github.com/facebookresearch/wav2letter).
If you want to use a language model, add `+criterion.wer_args='[/path/to/kenlm, /path/to/lexicon, 2, -1]'` to the command line.

### Evaluating a CTC model:

Evaluating a CTC model with a language model requires [flashlight python bindings](https://github.com/facebookresearch/flashlight/tree/master/bindings/python) (previously called [wav2letter](https://github.com/facebookresearch/wav2letter) to be installed.

Fairseq transformer language model used in the wav2vec 2.0 paper can be obtained from the [wav2letter model repository](https://github.com/facebookresearch/wav2letter/tree/master/recipes/sota/2019).
Be sure to upper-case the language model vocab after downloading it.

Letter dictionary for pre-trained models can be found [here](https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt).

Next, run the evaluation command:

```shell script
$subset=dev_other
python examples/speech_recognition/infer.py /checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw --task audio_finetuning \
--nbest 1 --path /path/to/model --gen-subset $subset --results-path /path/to/save/results/for/sclite --w2l-decoder kenlm \
--lm-model /path/to/kenlm.bin --lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
--post-process letter
```

To get raw numbers, use --w2l-decoder viterbi and omit the lexicon. To use the transformer language model, use --w2l-decoder fairseqlm.

## Use wav2vec 2.0 with 🤗Transformers:

Wav2Vec2 is also available in the [🤗Transformers library](https://github.com/huggingface/transformers) since version 4.4.

Pretrained Models can be found on the [hub](https://huggingface.co/models?filter=wav2vec2)
and documentation can be found [here](https://huggingface.co/transformers/master/model_doc/wav2vec2.html).

Usage example:

```python
# !pip install transformers
# !pip install datasets
import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# load pretrained model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


librispeech_samples_ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

# load audio
audio_input, sample_rate = sf.read(librispeech_samples_ds[0]["file"])

# pad input values and return pt tensor
input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values

# INFERENCE

# retrieve logits & take argmax
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)

# transcribe
transcription = processor.decode(predicted_ids[0])

# FINE-TUNE

target_transcription = "A MAN SAID TO THE UNIVERSE I EXIST"

# encode labels
with processor.as_target_processor():
  labels = processor(target_transcription, return_tensors="pt").input_ids

# compute loss by passing labels
loss = model(input_values, labels=labels).loss
loss.backward()
```

# wav2vec

Example to train a wav2vec model as described in [wav2vec: Unsupervised Pre-training for Speech Recognition (Schneider et al., 2019)](https://arxiv.org/abs/1904.05862).

## Pre-trained models

Description | Dataset | Model
---|---|---
Wav2Vec large | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt)

#### Example usage:
```python
import torch
import fairseq

cp_path = '/path/to/wav2vec.pt'
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
model = model[0]
model.eval()

wav_input_16khz = torch.randn(1,10000)
z = model.feature_extractor(wav_input_16khz)
c = model.feature_aggregator(z)
```

## Training a new model with the CLI tools

Given a directory containing wav files to be used for pretraining (we recommend splitting each file into separate files 10 to 30 seconds in length)

### Prepare training data manifest:

```
$ python examples/wav2vec/wav2vec_manifest.py /path/to/waves --dest /manifest/path --ext wav
```

### Train a wav2vec model:

```
$ python train.py /manifest/path --save-dir /model/path --num-workers 6 --fp16 --max-update 400000 --save-interval 1 --no-epoch-checkpoints \
--arch wav2vec --task audio_pretraining --min-lr 1e-06 --stop-min-lr 1e-09 --optimizer adam --lr 0.005 --lr-scheduler cosine \
--conv-feature-layers [(512, 10, 5), (512, 8, 4), (512, 4, 2), (512, 4, 2), (512, 4, 2), (512, 1, 1), (512, 1, 1)] \
--conv-aggregator-layers [(512, 2, 1), (512, 3, 1), (512, 4, 1), (512, 5, 1), (512, 6, 1), (512, 7, 1), (512, 8, 1), (512, 9, 1), (512, 10, 1), (512, 11, 1), (512, 12, 1), (512, 13, 1)] \
--skip-connections-agg --residual-scale 0.5 --log-compression --warmup-updates 500 --warmup-init-lr 1e-07 --criterion wav2vec --num-negatives 10 \
--max-sample-size 150000 --max-tokens 1500000 --skip-invalid-size-inputs-valid-test
```

### Run wav2vec2 pre-training on Google Cloud TPUs:

Wav2Vec2 is now supported on TPUs! It's currently pre-training only.

#### Using hydra on a v3-8:

```
$ OMP_NUM_THREADS=1 fairseq-hydra-train \
  task.data=/manifest/path \
  --config-dir /PATH/TO/FAIRSEQ/examples/wav2vec/config/pretraining \
  --config-name wav2vec2_large_librivox_tpu.yaml
```

#### Using command line arguments on a v3-8:
Note: Commandline arguments way of execution has a [known-problem](https://github.com/pytorch/fairseq/issues/3741) currently.

```
$ OMP_NUM_THREADS=1 python train.py /manifest/path --save-dir /model/path --num-workers 6 --fp16 --max-update 400000 --save-interval 1 --no-epoch-checkpoints \
--arch wav2vec2 --task audio_pretraining --min-lr 1e-06 --stop-min-lr 1e-09 --optimizer adam --lr 0.005 --lr-scheduler cosine \
--conv-feature-layers [(512, 10, 5), (512, 8, 4), (512, 4, 2), (512, 4, 2), (512, 4, 2), (512, 1, 1), (512, 1, 1)] \
--conv-aggregator-layers [(512, 2, 1), (512, 3, 1), (512, 4, 1), (512, 5, 1), (512, 6, 1), (512, 7, 1), (512, 8, 1), (512, 9, 1), (512, 10, 1), (512, 11, 1), (512, 12, 1), (512, 13, 1)] \
--skip-connections-agg --residual-scale 0.5 --log-compression --warmup-updates 500 --warmup-init-lr 1e-07 --criterion wav2vec --num-negatives 10 \
--max-sample-size 150000 --max-tokens 1500000 --skip-invalid-size-inputs-valid-test \
--tpu --distributed-world-size 8 --num-batch-buckets 3 --enable-padding \
--encoder-layerdrop 0 --mask-channel-prob 0.1
```

#### Using hydra on a pod slice (v3-N with N > 8):

```
$ OMP_NUM_THREADS=1 fairseq-hydra-train \
  task.data=/manifest/path \
  --config-dir /PATH/TO/FAIRSEQ/examples/wav2vec/config/pretraining \
  --config-name wav2vec2_large_librivox_tpu-pod.yaml  # edit distributed-world-size accordingly
```

#### Using command line arguments on a pod slice (v3-N with N > 8):
Note: Commandline arguments way of execution has a [known-problem](https://github.com/pytorch/fairseq/issues/3741) currently.

```
$ python -m torch_xla.distributed.xla_dist \
  --tpu ${TPUNAME} --conda-env=torch-xla-${TORCH_XLA_VERSION} --env OMP_NUM_THREADS=1 \
  -- \
python train.py /manifest/path --save-dir /model/path --num-workers 6 --fp16 --max-update 400000 --save-interval 1 --no-epoch-checkpoints \
--arch wav2vec2 --task audio_pretraining --min-lr 1e-06 --stop-min-lr 1e-09 --optimizer adam --lr 0.005 --lr-scheduler cosine \
--conv-feature-layers [(512, 10, 5), (512, 8, 4), (512, 4, 2), (512, 4, 2), (512, 4, 2), (512, 1, 1), (512, 1, 1)] \
--conv-aggregator-layers [(512, 2, 1), (512, 3, 1), (512, 4, 1), (512, 5, 1), (512, 6, 1), (512, 7, 1), (512, 8, 1), (512, 9, 1), (512, 10, 1), (512, 11, 1), (512, 12, 1), (512, 13, 1)] \
--skip-connections-agg --residual-scale 0.5 --log-compression --warmup-updates 500 --warmup-init-lr 1e-07 --criterion wav2vec --num-negatives 10 \
--max-sample-size 150000 --max-tokens 1500000 --skip-invalid-size-inputs-valid-test \
--tpu --distributed-world-size ${WORLD_SIZE} --num-batch-buckets 3 --enable-padding \
--encoder-layerdrop 0 --mask-channel-prob 0.1
```

### Extract embeddings from the downstream task data:

```
$ PYTHONPATH=/path/to/fairseq python examples/wav2vec/wav2vec_featurize.py --input /path/to/task/waves --output /path/to/output \
--model /model/path/checkpoint_best.pt --split train valid test
```

# vq-wav2vec

Example to train a vq-wav2vec model as described in [vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations (Baevski et al., 2019)](https://arxiv.org/abs/1910.05453).

These models are also used in [Effectiveness of self-supervised pre-training for speech recognition (Baevski et al., 2019)](https://arxiv.org/abs/1911.03912).

## Pre-trained models

Description | Dataset | Model
---|---|---
vq-wav2vec Gumbel | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec.pt)
vq-wav2vec K-means | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt)
Roberta on K-means codes | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/bert_kmeans.tar)

#### Example usage:
```python
import torch
import fairseq

cp = torch.load('/path/to/vq-wav2vec.pt')
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp])
model = model[0]
model.eval()

wav_input_16khz = torch.randn(1,10000)
z = model.feature_extractor(wav_input_16khz)
_, idxs = model.vector_quantizer.forward_idx(z)
print(idxs.shape) # output: torch.Size([1, 60, 2]), 60 timesteps with 2 indexes corresponding to 2 groups in the model
```

## Training a new model with the CLI tools

Given a directory containing wav files to be used for pretraining (we recommend splitting each file into separate file 10 to 30 seconds in length)

### Prepare training data manifest:

```
$ python examples/wav2vec/wav2vec_manifest.py /path/to/waves --dest /manifest/path --ext wav
```

### Train a gumbel vq-wav2vec model:

```
$ python train.py /manifest/path --save-dir /model/path --num-workers 6 --fp16 --max-update 400000 \
--save-interval 1 --no-epoch-checkpoints --arch wav2vec --task audio_pretraining --min-lr 1e-06 --stop-min-lr 1e-09 \
--optimizer adam --lr 1e-05 --lr-scheduler cosine \
--conv-feature-layers [(512, 10, 5), (512, 8, 4), (512, 4, 2), (512, 4, 2), (512, 4, 2), (512, 1, 1), (512, 1, 1), (512, 1, 1)] \
--conv-aggregator-layers [(512, 2, 1), (512, 3, 1), (512, 4, 1), (512, 5, 1), (512, 6, 1), (512, 7, 1), (512, 8, 1), (512, 9, 1), (512, 10, 1), (512, 11, 1), (512, 12, 1), (512, 13, 1)] \
--activation gelu --offset auto --skip-connections-agg --residual-scale 0.5 \
--log-keys ["prob_perplexity","code_perplexity","temp"] --vq-type gumbel --vq-groups 2 --vq-depth 2 \
--combine-groups --vq-vars 320 --vq-temp (2,0.5,0.999995) --prediction-steps 12 --warmup-updates 1000 \
--warmup-init-lr 1e-07 --criterion wav2vec --num-negatives 10 --max-sample-size 150000 \
--max-tokens 300000 --cross-sample-negatives 0 --update-freq 1 --seed 2 --skip-invalid-size-inputs-valid-test
```

for k-means training, set vq-type with "kmeans" and add --loss-weights [1] argument. Pre-trained models were trained on 16 GPUs.

### Tokenize audio data (e.g. for BERT training):

```
$ PYTHONPATH=/path/to/fairseq python examples/wav2vec/vq-wav2vec_featurize.py --data-dir /manifest/path --output-dir /path/to/output \
--checkpoint /model/path/checkpoint_best.pt --split train valid test --extension tsv
```
