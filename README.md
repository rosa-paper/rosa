# ROSA: Random Subspace Adaptation
This repository is the official implementation RoSA: Random Subspace Adaptation, a method for training large language models with limited memory.
## Requirements

To install requirements:

```setup
pip install -r requirements.txt
pip install -U datasets // update datasets library
pip install git+https://github.com/huggingface/transformers //
```

## Training

To train ROSA/LoRA model(s) on GLUE benchmark run this command:

```commandline
export DATASET_CACHE=/path/to/save/huggingface/dataset
export OUTPUT_PATH=/path/to/save/checkpoints
python train_mlm.py dataset.cache=$DATASET_CACHE output.path=$OUTPUT_PATH +task=cola model.name=roberta-base train.batch_size=16 train.lr=2e-5 fnmodel.name=rosa fnmodel.params.rank=8 fnmodel.params.factorize_method=svd_equal
```

## Evaluation
### Visualize train/validation curves of model(s)
Run the following command to visualize the train/validation curves of model(s) in the paper:

```commandline
tensorboard --logdir=/path/to/saved/runs
```

