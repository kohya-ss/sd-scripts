# BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension

[https://arxiv.org/abs/1910.13461](https://arxiv.org/abs/1910.13461)

## Introduction

BART is sequence-to-sequence model trained with denoising as pretraining objective. We show that this pretraining objective is more generic and show that we can match [RoBERTa](../roberta) results on SQuAD and GLUE and gain state-of-the-art results on summarization (XSum, CNN dataset), long form generative question answering (ELI5) and dialog response genration (ConvAI2). See the associated paper for more details.

## Pre-trained models

Model | Description | # params | Download
---|---|---|---
`bart.base` | BART model with 6 encoder and decoder layers | 140M | [bart.base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz)
`bart.large` | BART model with 12 encoder and decoder layers | 400M | [bart.large.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz)
`bart.large.mnli` | `bart.large` finetuned on `MNLI` | 400M | [bart.large.mnli.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz)
`bart.large.cnn` | `bart.large` finetuned on `CNN-DM` | 400M | [bart.large.cnn.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz)
`bart.large.xsum` | `bart.large` finetuned on `Xsum` | 400M | [bart.large.xsum.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.xsum.tar.gz)

## Results

**[GLUE (Wang et al., 2019)](https://gluebenchmark.com/)**
_(dev set, single model, single-task finetuning)_

Model | MNLI | QNLI | QQP | RTE | SST-2 | MRPC | CoLA | STS-B
---|---|---|---|---|---|---|---|---
`roberta.large` | 90.2 | 94.7 | 92.2 | 86.6 | 96.4 | 90.9 | 68.0 | 92.4
`bart.large` | 89.9 | 94.9 | 92.5 | 87.0 | 96.6 | 90.4 | 62.8 | 91.2

**[SQuAD (Rajpurkar et al., 2018)](https://rajpurkar.github.io/SQuAD-explorer/)**
_(dev set, no additional data used)_

Model | SQuAD 1.1 EM/F1 | SQuAD 2.0 EM/F1
---|---|---
`roberta.large` | 88.9/94.6 | 86.5/89.4
`bart.large` | 88.8/94.6 | 86.1/89.2

**[CNN/Daily Mail](http://nlpprogress.com/english/summarization.html)**
_(test set, no additional data used)_

Model | R1 | R2 | RL
---|---|---|---
`BERTSUMEXTABS` | 42.13 | 19.60 | 39.18
`bart.large` | 44.16 | 21.28 | 40.90

## Example usage

##### Load BART from torch.hub (PyTorch >= 1.1):
```python
import torch
bart = torch.hub.load('pytorch/fairseq', 'bart.large')
bart.eval()  # disable dropout (or leave in train mode to finetune)
```

##### Load BART (for PyTorch 1.0 or custom models):
```python
# Download bart.large model
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
tar -xzvf bart.large.tar.gz

# Load the model in fairseq
from fairseq.models.bart import BARTModel
bart = BARTModel.from_pretrained('/path/to/bart.large', checkpoint_file='model.pt')
bart.eval()  # disable dropout (or leave in train mode to finetune)
```

##### Apply Byte-Pair Encoding (BPE) to input text:
```python
tokens = bart.encode('Hello world!')
assert tokens.tolist() == [0, 31414, 232, 328, 2]
bart.decode(tokens)  # 'Hello world!'
```

##### Extract features from BART:
```python
# Extract the last layer's features
last_layer_features = bart.extract_features(tokens)
assert last_layer_features.size() == torch.Size([1, 5, 1024])

# Extract all layer's features from decoder (layer 0 is the embedding layer)
all_layers = bart.extract_features(tokens, return_all_hiddens=True)
assert len(all_layers) == 13
assert torch.all(all_layers[-1] == last_layer_features)
```

##### Use BART for sentence-pair classification tasks:
```python
# Download BART already finetuned for MNLI
bart = torch.hub.load('pytorch/fairseq', 'bart.large.mnli')
bart.eval()  # disable dropout for evaluation

# Encode a pair of sentences and make a prediction
tokens = bart.encode('BART is a seq2seq model.', 'BART is not sequence to sequence.')
bart.predict('mnli', tokens).argmax()  # 0: contradiction

# Encode another pair of sentences
tokens = bart.encode('BART is denoising autoencoder.', 'BART is version of autoencoder.')
bart.predict('mnli', tokens).argmax()  # 2: entailment
```

##### Register a new (randomly initialized) classification head:
```python
bart.register_classification_head('new_task', num_classes=3)
logprobs = bart.predict('new_task', tokens)
```

##### Batched prediction:
```python
import torch
from fairseq.data.data_utils import collate_tokens

bart = torch.hub.load('pytorch/fairseq', 'bart.large.mnli')
bart.eval()

batch_of_pairs = [
    ['BART is a seq2seq model.', 'BART is not sequence to sequence.'],
    ['BART is denoising autoencoder.', 'BART is version of autoencoder.'],
]

batch = collate_tokens(
    [bart.encode(pair[0], pair[1]) for pair in batch_of_pairs], pad_idx=1
)

logprobs = bart.predict('mnli', batch)
print(logprobs.argmax(dim=1))
# tensor([0, 2])
```

##### Using the GPU:
```python
bart.cuda()
bart.predict('new_task', tokens)
```

#### Filling masks:

BART can be used to fill multiple `<mask>` tokens in the input.
```python
bart = torch.hub.load('pytorch/fairseq', 'bart.base')
bart.eval()
bart.fill_mask(['The cat <mask> on the <mask>.'], topk=3, beam=10)
# [[('The cat was on the ground.', tensor(-0.6183)), ('The cat was on the floor.', tensor(-0.6798)), ('The cat sleeps on the couch.', tensor(-0.6830))]]
```

Note that by default we enforce the output length to match the input length.
This can be disabled by setting ``match_source_len=False``:
```
bart.fill_mask(['The cat <mask> on the <mask>.'], topk=3, beam=10, match_source_len=False)
# [[('The cat was on the ground.', tensor(-0.6185)), ('The cat was asleep on the couch.', tensor(-0.6276)), ('The cat was on the floor.', tensor(-0.6800))]]
```

Example code to fill masks for a batch of sentences using GPU
```
bart.cuda()
bart.fill_mask(['The cat <mask> on the <mask>.', 'The dog <mask> on the <mask>.'], topk=3, beam=10)
# [[('The cat was on the ground.', tensor(-0.6183)), ('The cat was on the floor.', tensor(-0.6798)), ('The cat sleeps on the couch.', tensor(-0.6830))], [('The dog was on the ground.', tensor(-0.6190)), ('The dog lay on the ground.', tensor(-0.6711)),
('The dog was asleep on the couch', tensor(-0.6796))]]
```

#### Evaluating the `bart.large.mnli` model:

Example python code snippet to evaluate accuracy on the MNLI `dev_matched` set.
```python
label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
ncorrect, nsamples = 0, 0
bart.cuda()
bart.eval()
with open('glue_data/MNLI/dev_matched.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[8], tokens[9], tokens[-1]
        tokens = bart.encode(sent1, sent2)
        prediction = bart.predict('mnli', tokens).argmax().item()
        prediction_label = label_map[prediction]
        ncorrect += int(prediction_label == target)
        nsamples += 1
        print('| Accuracy: ', float(ncorrect)/float(nsamples))
# Expected output: 0.9010
```

#### Evaluating the `bart.large.cnn` model:
- Follow instructions [here](https://github.com/abisee/cnn-dailymail) to download and process into data-files such that `test.source` and `test.target` has one line for each non-tokenized sample.
- For simpler preprocessing, you can also `wget https://cdn-datasets.huggingface.co/summarization/cnn_dm_v2.tgz`, although there is no guarantee of identical scores
- `huggingface/transformers` has a simpler interface that supports [single-gpu](https://github.com/huggingface/transformers/blob/master/examples/legacy/seq2seq/run_eval.py) and [multi-gpu](https://github.com/huggingface/transformers/blob/master/examples/legacy/seq2seq/run_distributed_eval.py) beam search.
    In `huggingface/transformers`, the BART models' paths are `facebook/bart-large-cnn` and `facebook/bart-large-xsum`.

In `fairseq`, summaries can be generated using:

```bash
cp data-bin/cnn_dm/dict.source.txt  checkpoints/
python examples/bart/summarize.py \
  --model-dir pytorch/fairseq \
  --model-file bart.large.cnn \
  --src cnn_dm/test.source \
  --out cnn_dm/test.hypo
```

For calculating rouge, install `files2rouge` from [here](https://github.com/pltrdy/files2rouge).

```bash
export CLASSPATH=/path/to/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar

# Tokenize hypothesis and target files.
cat test.hypo | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > test.hypo.tokenized
cat test.target | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > test.hypo.target
files2rouge test.hypo.tokenized test.hypo.target
# Expected output: (ROUGE-2 Average_F: 0.21238)
```


## Finetuning

- [Finetuning on GLUE](README.glue.md)
- [Finetuning on CNN-DM](README.summarization.md)

## Citation

```bibtex
@article{lewis2019bart,
    title = {BART: Denoising Sequence-to-Sequence Pre-training for Natural
Language Generation, Translation, and Comprehension},
    author = {Mike Lewis and Yinhan Liu and Naman Goyal and Marjan Ghazvininejad and
              Abdelrahman Mohamed and Omer Levy and Veselin Stoyanov
              and Luke Zettlemoyer },
    journal={arXiv preprint arXiv:1910.13461},
    year = {2019},
}
```
