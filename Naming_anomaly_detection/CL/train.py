import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
# from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.multiprocessing

import numpy as np
import matplotlib.pyplot as plt

from model import CLSingleHead
from huggingface import Huggingface
from loss import contrastive_loss
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer, 
                          DistilBertConfig, DistilBertModel, DistilBertTokenizer,
                          ConvBertConfig, ConvBertModel, ConvBertTokenizer,
                          BertConfig, BertModel, BertTokenizer,
                          AlbertConfig, AlbertModel, AlbertTokenizer,
                          ElectraConfig, ElectraModel, ElectraTokenizer,
                          DebertaConfig, DebertaModel, DebertaTokenizer,
                          MobileBertConfig, MobileBertModel, MobileBertTokenizer,
                          AutoConfig, AutoModel, AutoTokenizer,
                          )

from sklearn.metrics import classification_report

from tqdm import tqdm
import random

from loguru import logger
import os

PATH = 'multi_weights/RoBERTa_CL{}.pt'
MODELS = {
    'roberta': 'roberta-base',
    'bert': 'bert-base-uncased',
    'distilroberta': 'distilroberta-base',
    # 'distilbert': 'distilbert-base-uncased',
    'convbert': 'YituTech/conv-bert-base',
    # 'albert': 'albert-base-v2',
    'electra': 'google/electra-base-discriminator',
    'mobilebert': 'google/mobilebert-uncased',
    'tinybert': 'prajjwal1/bert-tiny', #T5? DEBERTA
    'deberta': 'microsoft/deberta-base',
    'longformer': 'allenai/longformer-base-4096',
    'bge': 'BAAI/bge-en-icl',
    'stella': 'dunzhang/stella_en_1.5B_v5'
    }
report_path = None
device = torch.device('cuda')
print(f"CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
torch.cuda.empty_cache()

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda", help='cuda or cpu')
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epoch', type=int, default=10, help='epochs')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
    parser.add_argument('--root', type=str, default='/depot/davisjam/data/mingyu/ptm-contrastive-learning/contrastive_learning', help='root directory for operation strings')
    parser.add_argument('--cp', '-checkpoint', type=str, default='', help='path to checkpoint of pretrained model')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    
    parser.add_argument("--output_dir_eval", default='./eval', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_dir_test", default='./test', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    parser.add_argument('--model_name', type=str, choices=['roberta', 'bert', 'distilroberta', 'distilbert', 'convbert', 'albert', 'electra', 'mobilebert', 'tinybert', 'deberta', 'longformer', 'bge', 'stella'], default='roberta', help='model type')
    parser.add_argument('--loss_fn', type=str, choices=['CL', 'CLCE', 'FoCL'], default='CLCE', help='Loss function')
    parser.add_argument('--lambd', type=float, default=0.3, help='lambda value for loss function, higher value adds more weight to CL loss')
    parser.add_argument('--trim', type=int, default=416, help='trim length (512 - trim_length) for RoBERTa')
    
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_eval', action='store_true', help='Whether to run eval on the eval set.')
    
    return parser.parse_args()

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def plot_loss_curve(epochs, train_loss, test_loss):
    epochs = np.arange(1, epochs + 1)
    logger.info(f"epochs: {epochs}")
    logger.info(f"train_loss: {train_loss}")
    logger.info(f"test_loss: {test_loss}")
    
    plt.figure()
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('multi_weights/plot/loss_curve.jpg')
    
def plot_f1_threshold_curve(f1_score, checkpoint_prefix, flag=None):
    threshold = f1_score.keys()

    plt.figure()
    plt.plot(threshold, f1_score.values(), marker='o', alpha=0.6, color='red')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Threshold Curve')
    plt.grid(True)
    if not flag:
        plt.savefig('multi_weights/plot/f1_threshold_curve_{}.jpg'.format(checkpoint_prefix))
    else:
        plt.savefig('multi_weights/plot/{}_f1_threshold_curve_{}.jpg'.format(flag, checkpoint_prefix))
    plt.close()
        
def plot_recall_precision_curve(precision, recall, checkpoint_prefix, flag=None):
    plt.figure()
    plt.plot(recall, precision, marker='o', alpha=0.6, color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs. Recall Curve')
    plt.grid(True)
    if not flag:
        plt.savefig('multi_weights/plot/precision_recall_curve_{}.jpg'.format(checkpoint_prefix))
    else:
        plt.savefig('multi_weights/plot/{}_precision_recall_curve_{}.jpg'.format(flag, checkpoint_prefix))
    plt.close()

def plot_roc_curve(tpr, fpr, checkpoint_prefix, flag=None):
    plt.figure()
    plt.plot(fpr, tpr, marker='o', alpha=0.6, color='green')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('TPR vs. FPR')
    plt.grid(True)
    if not flag:
        plt.savefig('multi_weights/plot/roc_curve_{}.jpg'.format(checkpoint_prefix))
    else:
        plt.savefig('multi_weights/plot/{}_roc_curve_{}.jpg'.format(flag, checkpoint_prefix))
    plt.close()
    
def train(args, tokenizer, model, optimizer, loss_fn):    
    #load Dataset
    train_dataset = Huggingface(args, root=args.root, tokenizer=tokenizer, trim=args.trim)
    train_loader = data.DataLoader(train_dataset, 
                                   batch_size=args.batch, 
                                   shuffle=True,
                                   num_workers=args.num_workers)
    
    logger.info(f"Training with model: {args.model_name}, lr: {args.lr}, batch_size: {args.batch}, epoch: {args.epoch}, loss_fn:{args.loss_fn}, lambd:{args.lambd}")
    
    train_loss, eval_loss = [], []
    
    for epoch in range(args.epoch): 
        logger.info(f"====epoch #{epoch+1}====")
    
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch[0]['input_ids'].to(args.device)
            attention_mask = batch[0]['attention_mask'].to(args.device)

            tokens = {'input_ids': input_ids, 'attention_mask': attention_mask}
            y_true = batch[1].to(args.device)
            layer_embeds, y_pred = model(**tokens)
            loss = loss_fn(layer_embeds, y_true, y_pred)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        train_loss.append(avg_loss)
        logger.info(f"Epoch {epoch + 1} Average Loss: {avg_loss:.5f}")
        
        # Run eval
        checkpoint_prefix = f'{args.model_name}_CL{epoch+1}' 
        
        epoch_eval_loss = eval(args, tokenizer, model, loss_fn, checkpoint_prefix) # change this to val_loss
        eval_loss.append(epoch_eval_loss)
        '''
        save checkpoint
        '''
        # if (epoch + 1) == 10:
        #     epoch_path = PATH.format(epoch+1)
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'rng_state': torch.random.get_rng_state(),
        #         }, epoch_path)        # divide up the PATH per epoch
        
        # plot_loss_curve(epoch+1, train_loss, test_loss)
        logger.info(f"epochs: {epoch+1}")
        logger.info(f"train_loss: {train_loss}")
        logger.info(f"test_loss: {eval_loss}")

    logger.info(f"Train loss: {train_loss}, Test loss: {eval_loss}")

def eval(args, tokenizer, model, loss_fn, checkpoint_prefix='RoBERTa_CL5'):
    # load dataset
    val_dataset = Huggingface(args, root=args.root, tokenizer=tokenizer, test=True, trim=args.trim)
    val_loader = data.DataLoader(val_dataset, 
                                 batch_size=args.eval_batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers)
    
    logger.info(f"***** Running evaluation {checkpoint_prefix}*****")
    logger.info(f"  Num examples = {len(val_dataset)}", )
    logger.info(f"  Batch size = {args.eval_batch_size}")
    # epoch = int(checkpoint_prefix[-1])
    model.eval()
    
    val_loss = 0.0
    val_acc = 0.0
    y_true, y_pred = [], []
    # all_embeddings = []
    # all_metadata = []
    
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
        input_ids = batch[0]['input_ids'].to(args.device)
        attention_mask = batch[0]['attention_mask'].to(args.device)
        tokens = {'input_ids': input_ids, 'attention_mask': attention_mask}
        y_true_ = batch[1].to(args.device)
        y_pred_ = None

        with torch.no_grad():
            layer_embeds, y_pred_ = model(**tokens)
        loss = loss_fn(layer_embeds, y_true_, y_pred_)
        val_loss += loss.item()
        # x.view(x.size(0), -1).
        # layer_embeds = layer_embeds[:,0,:]
        # all_embeddings.append(layer_embeds.view(layer_embeds.size(0), -1).detach().cpu().numpy())
        # all_metadata.extend([(val_dataset.reverse_model_type_class_dict[i], val_dataset.reverse_task_class_dict[j]) for i, j in y_true_.detach().cpu().numpy()])
        # correct_label = y_true_
        # correct_label = correct_label.detach().cpu().numpy()
        
        pred = torch.log_softmax(y_pred_, dim=1)
        _, predicted = torch.max(pred.data, 1)
        y_true.extend(y_true_.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        val_acc += (predicted == y_true_).sum().item()
    # all_embeddings = np.vstack(all_embeddings)
    
    val_loss /= len(val_loader)
    # if epoch %10 == 0 or epoch == 10:
    #     writer = SummaryWriter(f'runs/grid_search')
    #     # writer.add_image('image', np.ones((3,3,3)), 0); 
    #     writer.add_embedding(
    #         all_embeddings,
    #         metadata=all_metadata,
    #         metadata_header=['model_type', 'task'],
    #         # global_step=epoch,
    #         tag=f'{args.model_name}_{args.trim}'
    #     )
    #     writer.flush()
    #     writer.close()
    
    with open(report_path, 'a') as f:
        f.write(f"\n*****************{checkpoint_prefix}*****************\n")
    target_names = list(val_dataset.label_to_index.keys())
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}\n")
    labels = list(val_dataset.label_to_index.values())
    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=np.nan, digits=4)
    logger.info(f"Classification Report for {args.train_mode}\n{report}")
    logger.info(f"Validation Accuracy for {args.train_mode}: {round(val_acc / len(val_dataset), 4)}")  #fix accuracy
    with open(report_path, 'a') as f:
        f.write(report)
    
    logger.info(f"Classification report saved to {report_path}")
    logger.info(f"Validation loss: {val_loss}")
    
    return val_loss

def add_tokens(model, tokenizer):
    filepath = "/depot/davisjam/data/mingyu/ptm-contrastive-learning/contrastive_learning/PeaTMOSS_vocab.txt"
    with open(filepath, 'r') as f:
        file = f.read().split('\n')
        operations = [op for op in file if op is not None and isinstance(op, str)]
    num_added_toks = tokenizer.add_tokens(operations)
    logger.info(f"We have added {num_added_toks} tokens")
    model.resize_token_embeddings(len(tokenizer))
    
def main():
    args = parse_arg()
    set_seed()
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model_name], use_fast=False, cache_dir='/scratch/gilbreth/kim3118/.cache/huggingface')
    model = AutoModel.from_pretrained(MODELS[args.model_name], cache_dir='/scratch/gilbreth/kim3118/.cache/huggingface')
    add_tokens(model, tokenizer)
    
    global report_path
    
    report_path = os.path.join(args.output_dir_eval, f"{args.model_name}_{'multi' if args.multi_label else args.train_mode}_{args.loss_fn}_{args.batch}_{args.lr}_report")
    idx = 1
    while os.path.exists(report_path):
        if idx == 1:
            report_path += f"_{str(idx)}"
        else:
            report_path = report_path[:-len(str(idx-1))] + str(idx)
        idx += 1
        logger.debug(f"report path exists: {report_path}")

    train_dataset = Huggingface(args, root=args.root, tokenizer=tokenizer, trim=args.trim)
    tau = 50
    logger.info(f"trim: {args.trim}")
    logger.info(f"tau: {tau}")
    
    num_labels = len(train_dataset.label_to_index)
    model = CLSingleHead(args, model, num_labels).to(args.device)
    loss_fn = contrastive_loss(args, tau)    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer.zero_grad()
            
    if args.do_train:
        result = train(args, tokenizer, model, optimizer, loss_fn)
    if args.do_eval:
        # checkpoint_prefix = 'RoBERTa_CL8'
        if not os.path.exists(args.output_dir_eval):
            os.makedirs(args.output_dir_eval)
        result = eval(args, tokenizer, model, loss_fn)
        
    return result

if __name__ == '__main__':
    main() 