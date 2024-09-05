# Contrastive Learning Solution

Train and evaluate contrastive learning solution for analyzing defects in PTM repositories

## Example Usage (Longformer)

python train.py --do_train --train_mode='model_type' --loss_fn='CLCE' --model_name='longformer' --batch=32 --eval_batch_size=32 --lr=5e-5 --epoch=30 --lambd=0.3

## Example Usage (RoBERTa)

python train_prev.py --do_train --train_mode='model_type' --loss_fn='CLCE' --model_name='roberta' --batch=64 --eval_batch_size=64 --lr=5e-5 --epoch=40 --trim=416 --lambd=0.3


## Parameters

Explain the main parameters:

- `--do_train`: Run training
- `--train_mode`: Choose from 'model_type', 'task', or 'arch'
- `--loss_fn`: Loss function (CLCE, CL, FoCL)
- `--model_name`: Name of the model to use
- `--batch`: Training batch size
- `--eval_batch_size`: Evaluation batch size
- `--lr`: Learning rate
- `--epoch`: Number of training epochs
- `--lambd`: Lambda value for loss function
- `--trim`: Trim length for input (RoBERTa-specific)

## Project Structure

- `train.py`: Main training script
- `model.py`: Model architecture definition
- `loss.py`: Loss function implementations
- `huggingface.py`: Dataset loading and preprocessing
- `PeaTMOSS_data/*`: Train and evaluation dataset

## Results

### Longformer Accuracies
- Model Type: 87.67%
- Task: 50.37%
- Architecture: 50.66%

### RoBERTa Accuracies
- Model Type: 85.81%
- Task: 46.40%
- Architecture: 49.34%
