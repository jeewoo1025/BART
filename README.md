# BART
BART fine-tuning 
<br>

## Description of codes
* `train.py` : train and valid 
* `inference.py` : eval and make summarization on test dataset
* summarizer folder
  * `dataloader.py` : dataloader of xsum and cnn/dm dataset
  * `default.py` : summarization model
  * `scheduler.py` : learning rate scheduler
  * `utils.py` : logger function

### Workspace
Following directories should be created 
* `./outputs` : store model checkpoints

<br>

## Dataset
* <strong>XSum</strong>
using https://github.com/EdinburghNLP/XSum dataset or download [preprocessed data](https://github.com/yixinL7/BRIO)
<br>

## How to Run
I run codes on `NVIDIA GeForce RTX 3090`. It takes 10 ~ 13 hours for 1 epoch.

### Train
```python
python train.py --dataset-path [xsum or cnn/dm dataset] --output-dir [output dir path] --epochs 10 --max-learning-rate 2e-3 --batch-size 4 --valid-batch-size 8
```

### Inference
```python
python inference.py --pretrained-ckpt-path [fine-tuned model path] --dataset-path [xsum or cnn/dm dataset] --output-path [saved result path]
```
