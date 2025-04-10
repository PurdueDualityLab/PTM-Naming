import argparse
import json
import os
import random
from collections import defaultdict
import time

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from dotenv import load_dotenv
from loguru import logger
from sklearn.metrics import confusion_matrix, average_precision_score, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    DataCollatorForLanguageModeling,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
    AutoConfig, 
    AutoModel, 
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback
)

from DARA.utils import eval_metrics, plot_accuracy, plot_loss, top_k_predictions
from dataloader import transformer_dataset
from loss import contrastive_loss
from cl_trainer import CustomCLTrainer

load_dotenv()
# os.environ['HF_HOME'] = os.getenv("HF_HOME")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

MODELS = {
    'roberta': 'roberta-base',
    'bert': 'bert-base-uncased',
    'distilroberta': 'distilroberta-base',
    'convbert': 'YituTech/conv-bert-base',
    'electra': 'google/electra-base-discriminator',
    'mobilebert': 'google/mobilebert-uncased',
    'tinybert': 'prajjwal1/bert-tiny', 
    'deberta': 'microsoft/deberta-base',
    'longformer': 'allenai/longformer-base-4096',
    'bge': 'BAAI/bge-en-icl',
    'stella': 'dunzhang/stella_en_1.5B_v5'
    }

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epoch', type=int, default=30, help='epochs')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
    parser.add_argument('--root', type=str, default='/depot/davisjam/data/mingyu/ptm-contrastive-learning/contrastive_learning', help='root directory for operation strings')
    parser.add_argument('--cp', '-checkpoint', type=str, default='', help='path to checkpoint of pretrained model')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    
    parser.add_argument("--output_dir", default='./layer_pretrained', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    parser.add_argument('--model_name', type=str, choices=['roberta', 'bert', 'distilroberta', 'distilbert', 'convbert', 'albert', 'electra', 'mobilebert', 'tinybert', 'deberta', 'longformer', 'bge', 'stella'], default='roberta', help='model type')
    parser.add_argument('--loss_fn', type=str, choices=['CL', 'CLCE', 'FoCL'], default='CLCE', help='Loss function')
    parser.add_argument('--tau', type=float, default=0.3, help='tau value for loss function, higher values soften the similarity scores')
    parser.add_argument('--lambd', type=float, default=0.1, help='lambda value for loss function, higher value adds more weight to CL loss')
    parser.add_argument('--trim', type=int, default=416, help='trim length (512 - trim_length) for RoBERTa')
    
    parser.add_argument('--train_mode', type=str, choices=['pre-train', 'fine-tune'], default='fine-tune', help='train mode')
    # Sub-options for pre-training
    parser.add_argument(
        '--pretrain_type', type=str, choices=['full', 'continued'], 
        default='continued', help='Choose between full pre-training (from scratch) or domain-adaptive pretraining (continued training)'
    )
    # Sub-options for fine-tuning
    parser.add_argument(
        '--finetune_type', type=str, choices=['cross-entropy', 'contrastive'], 
        default='cross-entropy', help='Choose between using cross-entropy or contrastive learning with cross-entropy for fine-tuning'
    )
    
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_eval', action='store_true', help='Whether to run eval on the eval set.')
    
    
    parser.add_argument('--label_type', type=str, default="model_type", help='label_type')
    parser.add_argument('--top_k', type=int, default=2, help='if label_type is task, then use multi-label classification')
    
    # parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    return parser.parse_args()

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    
def pre_train(args, tokenizer, train_dataset, eval_dataset):
    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    training_args = TrainingArguments(
        output_dir=args.output_dir + '_' + args.model_name,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=10,  
        weight_decay=0.01,
        save_total_limit=5,
        load_best_model_at_end=True,
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    best_model_path = training_args.output_dir  # The best model is saved here
    model.save_pretrained(best_model_path)
    return best_model_path
    

def fine_tune(args, model, index_to_label, train_dataset, eval_dataset, output_dir):
    if args.model_name == 'longformer' and args.label_type == 'arch':
        gradient_accumulation_steps = 64 // 8
        logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    elif args.model_name == 'longformer' and args.label_type == 'model_type':
        gradient_accumulation_steps = 32 // 8
        logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    else:
        gradient_accumulation_steps = 1

    training_args = TrainingArguments(
        # output_dir=output_dir,
        output_dir=None,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        dataloader_drop_last=False,
        metric_for_best_model="eval_loss",
        gradient_accumulation_steps = gradient_accumulation_steps
    )

    def compute_metrics(eval_pred):
        """
        Computes evaluation metrics, handling both multi-class and multi-label classification,
        and optionally calculates top-k accuracy for multi-label.

        Args:
            eval_pred: A tuple (logits, labels) where logits are model outputs and labels are true labels.
            top_k: If provided, calculates top-k accuracy for multi-label.
        """

        logits, labels = eval_pred
        print(f"Type of eval_pred: {type(eval_pred)}")
        print(f"Type of logits: {type(logits)}")
        print(f"Type of labels: {type(labels)}")
        print(f"Shape of logits: {logits.shape}")
        print(f"Shape of labels: {labels.shape}")

        if args.label_type == 'task':
            # Multi-label classification (sigmoid and threshold)
            predictions_1 = top_k_predictions(logits, 1)
            predictions_2 = top_k_predictions(logits, 2)
            predictions_3 = top_k_predictions(logits, 3)
            eval_1 = eval_metrics(labels, predictions_1, 1)
            eval_2 = eval_metrics(labels, predictions_2, 2)
            eval_3 = eval_metrics(labels, predictions_3, 3)
            combined_results = {**eval_1, **eval_2, **eval_3}
            return combined_results
            # if args.top_k is not None:
                # Multi-label with top-k predictions
            # else:
            #     # Standard multi-label (sigmoid and threshold)
            #     predictions = (np.sigmoid(logits) > 0.5).astype(int)
        else:
            # Multi-class classification (argmax)
            predictions = np.argmax(logits, axis=-1)
            return eval_metrics(labels, predictions)
        
    if args.finetune_type == 'contrastive':
        loss_fn = contrastive_loss(device, args.loss_fn, args.tau, args.lambd, args.label_type)
        trainer = CustomCLTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            loss_fn=loss_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )
    train_start_time = time.time()
    trainer.train()
    train_end_time = time.time()
    T_train = train_end_time - train_start_time
    logger.info(f"Training time: {T_train:.2f} seconds")
    
    predict_start_time = time.time()
    predictions = trainer.predict(eval_dataset)
    predict_end_time = time.time()
    T_predict = predict_end_time - predict_start_time
    logger.info(f"Prediction time: {T_predict:.2f} seconds")
    
    logits = predictions.predictions
    labels = predictions.label_ids
    # if args.label_type == 'task':
    #     if args.top_k is not None:
    #         predicted_labels = top_k_predictions(logits, args.top_k)
    #     else:
    #         predicted_labels = (np.sigmoid(logits) > 0.5).astype(int)
    # else:
    #     predicted_labels = np.argmax(logits, axis=-1)

    eval_results = defaultdict(list)
    train_losses = []
    eval_losses = []
    
    for log in trainer.state.log_history:
        if 'loss' in log:
            train_losses.append(log['loss'])
        if 'eval_loss' in log:
            eval_losses.append(log['eval_loss'])
            for k, v in log.items():
                if k.startswith('eval_'):
                    eval_results[k[5:]].append(v)

    preds = []
    targets = []

    # for pred_idx, target_idx in zip(predicted_labels, labels):
    #     if isinstance(pred_idx, np.ndarray):
    #         # Multi-label prediction (assuming pred_idx is an array of indices)
    #         pred_labels = [index_to_label.get(i) for i in pred_idx if i in index_to_label]
    #         preds.append(pred_labels)
    #     else:
    #         # Single-label prediction
    #         preds.append(index_to_label.get(pred_idx))

    #     if isinstance(target_idx, np.ndarray):
    #         # Multi-label target (assuming target_idx is an array of indices)
    #         target_labels = [index_to_label.get(i) for i in target_idx if i in index_to_label]
    #         targets.append(target_labels)
    #     else:
    #         # Single-label target
    #         targets.append(index_to_label.get(target_idx))
    # preds = [eval_dataset.index_to_label[idx] for idx in predicted_labels]
    # targets = [eval_dataset.index_to_label[idx] for idx in labels]
    
    # eval_results = eval_metrics(labels, predicted_labels, None if args.label_type != 'task' else args.top_k)
    print(f"Eval results: {eval_results}")
    return train_losses, eval_losses, eval_results, preds, targets

# maybe do CV run?
def CV_run():
    ############################
    # hyperparameters

    # label_type = "model_type"
    # label_type = "task"
    # label_type = "arch"
    ############################
    args = parse_arg()
    # if args.local_rank != -1:
    #     torch.cuda.set_device(args.local_rank)
    #     device = torch.device("cuda", args.local_rank)
    #     torch.distributed.init_process_group(backend='nccl', init_method='env://')
    #     logger.info(f"Using distributed training on GPU: {args.local_rank}")
    # else:
    #     logger.info(f"Using single GPU: {device}")
        
    if args.label_type == "task":
        vec_path = 'Naming_anomaly_detection/DARA/transformer/data/data_cleaned.json'
        logger.info(f"Using multi-label classification")
    else:
        vec_path = 'Naming_anomaly_detection/DARA/transformer/data/data.json'
        args.top_k = None
    # vec_path = 'data_cleaned.json'
    
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model_name])
    
    full_dataset = transformer_dataset(dict_path=vec_path, label_type=args.label_type, tokenizer=tokenizer, trim=args.trim)

    id2label = full_dataset.get_label_mapping()
    label2id = full_dataset.get_index_mapping()
    # index_to_label = full_dataset.get_label_mapping()
    num_labels = full_dataset.get_num_classes()
    # model = AutoModelForSequenceClassification.from_pretrained(MODELS[args.model_name], num_labels=num_labels)
    print(f"Length of full_dataset: {len(full_dataset)}")
    logger.info(f"Number of classes: {num_labels}")
    logger.info(f"label type: {args.label_type}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Variables to store cumulative results
    cumulative_train_losses, cumulative_eval_losses = [], []
    cumulative_eval_metrics = defaultdict(list)


    fold = 0  # Counter for current fold

    all_fold_preds = []
    all_fold_targets = []
    
    if args.train_mode == 'pre-train':
        print(f"Train mode: {args.train_mode}")
        print(f"Pre-train type: {args.pretrain_type}")
    elif args.train_mode == 'fine-tune':
        print(f"Train mode: {args.train_mode}")
        print(f"Fine-tune type: {args.finetune_type}")
        
    print(f"lr: {args.lr}")
    print(f"batch size: {args.batch_size}")
    train_type = args.finetune_type if args.train_mode == 'fine-tune' else args.pretrain_type
    output_dir = f"{args.model_name}_{args.train_mode}_{train_type}_{args.label_type}_{args.lr}_{args.batch_size}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for train_index, eval_index in kf.split(full_dataset):
        fold += 1
        logger.info(f"Starting fold {fold}")

        # Create datasets for the current fold
        train_dataset = torch.utils.data.Subset(full_dataset, train_index)
        eval_dataset = torch.utils.data.Subset(full_dataset, eval_index)

        # # Initialize DataLoaders for the current fold
        # train_loader = DataLoader(train_subset, batch_size=train_batch_size, shuffle=True, num_workers=0, pin_memory=True)
        # eval_loader = DataLoader(eval_subset, batch_size=eval_batch_size, num_workers=0, pin_memory=True)

        # if fold > 1:
        #     continue    #for quick prototype

        # Initialize the model for the current fold
        if args.train_mode == 'pre-train':
            if args.pretrain_type == 'full':
                logger.info("Starting full pre-training (training from scratch)")
                
                tokenizer = ByteLevelBPETokenizer(lowercase=True)
                tokenizer.train(
                    files=["training_corpus.txt"],
                    vocab_size=300,         # Set slightly above your distinct tokens (173) for flexibility
                    min_frequency=1,        # Ensure even tokens with minimal occurrences are included
                    show_progress=True,
                    special_tokens=[
                        "<s>",
                        "<pad>",
                        "</s>",
                        "<unk>",
                        "<mask>",
                    ]
                )
            else:
                # TODO: add tokens and resize token embeddings
                # model.resize_token_embeddings(len(tokenizer))
                logger.info("Starting domain-adaptive pre-training (continued pre-training)")
            best_model_path = pre_train(args, tokenizer, train_dataset, eval_dataset)
            model = AutoModelForSequenceClassification.from_pretrained(best_model_path, num_labels=num_labels)
            args.epoch = 20
        if args.train_mode == 'fine-tune':
            if args.label_type == 'task':   # multi label classification
                model = AutoModelForSequenceClassification.from_pretrained(
                        MODELS[args.model_name], 
                        problem_type="multi_label_classification",
                        num_labels=num_labels, 
                        id2label=id2label,
                        label2id=label2id
                    )
            else:   # single label classification (multi-class)
                model = AutoModelForSequenceClassification.from_pretrained(
                        MODELS[args.model_name], 
                        num_labels=num_labels,
                        id2label=id2label,
                        label2id=label2id
                    )
            
        if args.finetune_type == 'cross-entropy':   # Cross-entropy loss
            logger.info("Starting fine-tuning with cross-entropy loss")
        else:   # Contrastive learning with cross-entropy loss
            # args.batch_size = 128
            # args.eval_batch_size = 32
            logger.info("Starting fine-tuning with combined contrastive learning & cross-entropy loss")
            logger.info(f"Loss function: {args.loss_fn}")
            logger.info(f"tau: {args.tau}")
            logger.info(f"lambda: {args.lambd}")
        train_losses, eval_losses, eval_results, preds, targets = fine_tune(args, model, id2label, train_dataset, eval_dataset, output_dir)
        
        logger.info(f'Fold {fold}, Final Eval Metrics: {", ".join([f"{k}: {v[-1]:.2f}" for k, v in eval_results.items()])}')
            
        all_fold_preds.extend(preds)
        all_fold_targets.extend(targets)
        cumulative_train_losses.append(train_losses)
        cumulative_eval_losses.append(eval_losses)
        for key, value in eval_results.items():
            cumulative_eval_metrics[key].append(value)
        # if args.model_name == 'longformer': # only running first fold
        #     break
    print(cumulative_eval_metrics['accuracy'])

    average_train_loss = [sum(losses) / len(losses) for losses in zip(*cumulative_train_losses)]
    average_eval_loss = [sum(losses) / len(losses) for losses in zip(*cumulative_eval_losses)]
    # average_eval_accuracy = [sum(accs) / len(accs) for accs in zip(*cumulative_eval_metrics['accuracy'])]
    # logger.info(f"Average Eval Accuracy across all folds: {average_eval_accuracy[-1]:.2f}%")
    if args.train_mode == 'fine-tune':
        root_dir = f'Naming_anomaly_detection/DARA/transformer/{args.model_name}_{args.finetune_type}'
    else:
        root_dir = f'Naming_anomaly_detection/DARA/transformer/{args.model_name}_{args.pretrain_type}_{args.finetune_type}'
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)
    results_dir = os.path.join(root_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    logger.success("5-Fold Cross Validation completed")
    for k, v in cumulative_eval_metrics.items():
        last_element = [sublist[-1] for sublist in v]
        mean = np.mean(last_element)
        std = np.std(last_element)
        print(f"{k}: {mean:.4f} (± {std:.4f})")
        
    plot_loss(average_train_loss, average_eval_loss, args.epoch, args.lr, args.batch_size, args.label_type, eval_interval=1, root_dir=root_dir) 
    # plot_accuracy(average_eval_accuracy, args.epoch, args.lr, args.batch_size, args.label_type, root_dir)
        
    # np.save(f'{root_dir}/results/preds_final_{output_dir}.npy', all_fold_preds)
    # np.save(f'{root_dir}/results/targets_final_{output_dir}.npy', all_fold_targets)
    with open(f'{root_dir}/results/{args.label_type}_index_to_label_final.json', 'w') as f:
        json.dump(id2label, f)

if __name__ == '__main__':
    set_seed(0)
    CV_run()
