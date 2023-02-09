# Finetuning RoBERTa on a custom classification task

This example shows how to finetune RoBERTa on the IMDB dataset, but should illustrate the process for most classification tasks.

### 1) Get the data

```bash
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar zxvf aclImdb_v1.tar.gz
```


### 2) Format data

`IMDB` data has one data-sample in each file, below python code-snippet converts it one file for train and valid each for ease of processing.  
```python
import argparse
import os
import random
from glob import glob

random.seed(0)

def main(args):
    for split in ['train', 'test']:
        samples = []
        for class_label in ['pos', 'neg']:
            fnames = glob(os.path.join(args.datadir, split, class_label) + '/*.txt')
            for fname in fnames:
                with open(fname) as fin:
                    line = fin.readline()
                    samples.append((line, 1 if class_label == 'pos' else 0))
        random.shuffle(samples)
        out_fname = 'train' if split == 'train' else 'dev'
        f1 = open(os.path.join(args.datadir, out_fname + '.input0'), 'w')
        f2 = open(os.path.join(args.datadir, out_fname + '.label'), 'w')
        for sample in samples:
            f1.write(sample[0] + '\n')
            f2.write(str(sample[1]) + '\n')
        f1.close()
        f2.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='aclImdb')
    args = parser.parse_args()
    main(args)
```


### 3) BPE encode

Run `multiprocessing_bpe_encoder`, you can also do this in previous step for each sample but that might be slower.
```bash
# Download encoder.json and vocab.bpe
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'

for SPLIT in train dev; do
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json encoder.json \
        --vocab-bpe vocab.bpe \
        --inputs "aclImdb/$SPLIT.input0" \
        --outputs "aclImdb/$SPLIT.input0.bpe" \
        --workers 60 \
        --keep-empty
done
```


### 4) Preprocess data

```bash
# Download fairseq dictionary.
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'  

fairseq-preprocess \
    --only-source \
    --trainpref "aclImdb/train.input0.bpe" \
    --validpref "aclImdb/dev.input0.bpe" \
    --destdir "IMDB-bin/input0" \
    --workers 60 \
    --srcdict dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "aclImdb/train.label" \
    --validpref "aclImdb/dev.label" \
    --destdir "IMDB-bin/label" \
    --workers 60

```


### 5) Run training

```bash
TOTAL_NUM_UPDATES=7812  # 10 epochs through IMDB for bsz 32
WARMUP_UPDATES=469      # 6 percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler.
HEAD_NAME=imdb_head     # Custom name for the classification head.
NUM_CLASSES=2           # Number of classes for the classification task.
MAX_SENTENCES=8         # Batch size.
ROBERTA_PATH=/path/to/roberta.large/model.pt

CUDA_VISIBLE_DEVICES=0 fairseq-train IMDB-bin/ \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_large \
    --criterion sentence_prediction \
    --classification-head-name $HEAD_NAME \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 10 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --find-unused-parameters \
    --update-freq 4
```

The above command will finetune RoBERTa-large with an effective batch-size of 32
sentences (`--batch-size=8 --update-freq=4`). The expected
`best-validation-accuracy` after 10 epochs is ~96.5%.

If you run out of GPU memory, try decreasing `--batch-size` and increase
`--update-freq` to compensate.


### 6) Load model using hub interface

Now we can load the trained model checkpoint using the RoBERTa hub interface.

Assuming your checkpoints are stored in `checkpoints/`:
```python
from fairseq.models.roberta import RobertaModel
roberta = RobertaModel.from_pretrained(
    'checkpoints',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='IMDB-bin'
)
roberta.eval()  # disable dropout
```

Finally you can make predictions using the `imdb_head` (or whatever you set
`--classification-head-name` to during training):
```python
label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)

tokens = roberta.encode('Best movie this year')
pred = label_fn(roberta.predict('imdb_head', tokens).argmax().item())
assert pred == '1'  # positive

tokens = roberta.encode('Worst movie ever')
pred = label_fn(roberta.predict('imdb_head', tokens).argmax().item())
assert pred == '0'  # negative
```
