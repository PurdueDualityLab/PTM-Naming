# Overview

This folder includes all the scripts and data used in our contrastive learning pipeline (CL, RQ4).

## Example Usage (Longformer)

python Naming_anomaly_detection/DARA/transformer/run.py --lr=0.00005 --label_type=model_type --model_name=longformer --batch_size=8 --epoch=50 --finetune_type=contrastive --lambd=0.1 --tau=0.1

## Example Usage (RoBERTa)

python Naming_anomaly_detection/DARA/transformer/run.py --lr=0.00005 --label_type=model_type --model_name=roberta --batch_size=256 --epoch=50 --finetune_type=contrastive --lambd=0.1 --tau=0.1


## Parameters


- `--label_type`: Choose from `model_type`, `task`, or `arch`
- `--loss_fn`: Loss function (SupConCL)
- `--model_name`: Name of the model to use
- `--batch`: Training batch size
- `--train_mode`: Choose from `pre-train` or `fine-tune`
- `--finetune_type`: Suboptions for fine-tuning (Choose from `cross-entropy` or `contrastive`)
- `--eval_batch_size`: Evaluation batch size
- `--lr`: Learning rate
- `--epoch`: Number of training epochs
- `--lambd`: Lambda value for loss function
- `--trim`: Trim length for input (RoBERTa-specific)

## Project Structure

- `run.py`: Main training script
- `loss.py`: Loss function implementations
- `dataloader.py`: Dataset loading
- `data_pre.py`: Dataset preprocessing
- `data/*`: Full dataset
- `cl_trainer.py`: Custom `trainer` wrapper

