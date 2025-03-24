import os
import random
import subprocess
from loguru import logger
from dotenv import load_dotenv
import json
import time

load_dotenv()
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from dataloader import DARA_dataset
BASE_DIR = '/depot/davisjam/data/mingyu/PTM-Naming'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PEATMOSS_PATH = os.getenv("PEATMOSS_VEC_DATA_PATH")
json_file_loc_flop = os.path.join(PEATMOSS_PATH, "flop.json")
if os.path.exists(json_file_loc_flop):
    with open(json_file_loc_flop, 'r') as f:
        params = json.load(f)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def run_all_models(model_list, label_type, fold):
    for repo in model_list:
        if repo in params:
            print(f"{repo} already exists, skipping...")
            continue
        exit()
        print(f"Starting evaluation for model: {repo}")
        p = subprocess.Popen(["python", "Naming_anomaly_detection/DARA/latency_by_one.py", "--repo", repo, "--label_type", label_type, "--fold", str(fold)])    
        p.wait()

def CV_run():
    
    # ############################
    # # hyperparameters
    # epochs = 40
    # lr = 1e-3
    # train_batch_size = 256
    eval_batch_size = 32

    label_type = "model_type"
    # label_type = "arch" # TODO: This needs a different hyperparameter setting
    # label_type = "task"
    ############################
    vec_path = './data_cleaned.json'
    # vec_path = './data_cleaned_filtered.json'
    
    # data_loader = DataLoader(vec_path)

    full_dataset = DARA_dataset(dict_path=vec_path, label_type=label_type)

    index_to_label = full_dataset.get_label_mapping()
    num_classes = full_dataset.get_num_classes()
    input_shape = full_dataset.get_data_shape()
    
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"label type: {label_type}")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold = 0  # Counter for current fold

    for train_index, eval_index in kf.split(full_dataset):
        fold += 1
        logger.info(f"Starting fold {fold}")

        # Create datasets for the current fold (simulating train-test split)
        eval_subset = torch.utils.data.Subset(full_dataset, eval_index)

        # Initialize DataLoaders for the current fold
        eval_loader = DataLoader(eval_subset, batch_size=eval_batch_size, num_workers=0, pin_memory=True)
        
        # # Initialize the model for the current fold
        # model = DARA_classifier(input_size=input_shape[1], output_size=num_classes)
        # PATH = f'{BASE_DIR}/fold_{fold}_model_state_dict.pt'
        # model.load_state_dict(torch.load(PATH))
        # model = model.to(device)
        # model.eval()

        # json_file_loc_latency = os.path.join(PEATMOSS_PATH, "latency.json")
        # os.makedirs(os.path.dirname(json_file_loc_latency), exist_ok=True)
        
        # json_file_loc_flop = os.path.join(PEATMOSS_PATH, "flop.json")
        # os.makedirs(os.path.dirname(json_file_loc_flop), exist_ok=True)


        with torch.no_grad():
            start_time = time.time()
            for _, target, names in eval_loader:
                run_all_models(names, label_type, fold)
            #     for i, name in enumerate(names):
            #         if os.path.exists(json_file_loc_latency):
            #             with open(json_file_loc_latency, 'r') as f:
            #                 latency = json.load(f)
            #         else:
            #             latency = {}
                        
            #         # if os.path.exists(json_file_loc_flop):
            #         #     with open(json_file_loc_flop, 'r') as f:
            #         #         flop = json.load(f)
            #         # else:
            #         #     flop = {}
            #         print(torch.cuda.memory_summary())
            #         if name in latency and latency[name] != -1:
            #             logger.info(f"Latency for {name} already exists.")
            #             continue
                    
            #         # create vector
            #         logger.info(f"Exporting APTM for {name}")
            #         sample_start_time = time.time()
            #         result = export_vector(name)
            #         # flops, macs, params = calculateFlop(name)
                    
            #         # failed to export vector
            #         latency[name] = -1
            #         if not result:
            #             with open(json_file_loc_latency, 'w') as f:
            #                 json.dump(latency, f)
            #             continue
            #         output_path, T_aptm, T_export, complexity = result
            #         total_aptm_time += T_aptm
            #         total_export_time += T_export
            #         output_path = result
            #         # Code for calculating FLOP
            #         # flops, macs, params = complexity
            #         # if flops != -1:
            #         #     flop[name] = {"flops": flops, "macs": macs, "params": params}
            #         # else:
            #         #     flop[name] = {"flops": -1, "macs": -1, "params": -1}
            #         # with open(json_file_loc_flop, "w") as f:
            #         #     json.dump(flop, f)
                        
            #         # data processing
            #         process_start_time = time.time()
            #         data = data_processing(data_path=output_path, label_type=label_type)
            #         T_process = time.time() - process_start_time
            #         total_process_time += T_process
                    
            #         data = data.to(device)
            #         inference_start_time = time.time()
            #         output = model(data)
                    

            #         _, predicted = torch.max(output, 1)
            #         T_inference = time.time() - inference_start_time
            #         total_inference_time += T_inference
            #         # logger.info(f'index_to_label: {index_to_label}, predict: {predicted.item()}, target: {target[i]}')
            #         logger.info(f'predict: {index_to_label[predicted.item()]}, target: {index_to_label[target[i].item()]}')
            #         T_total = time.time() - sample_start_time
            #         T_overhead = T_total - (T_aptm + T_export + T_process + T_inference)
            #         total_overhead_time += T_overhead
            #         # latency calculation
            #         latency[name] = {
            #             'aptm': T_aptm,
            #             'export': T_export,
            #             'process': T_process,
            #             'inference': T_inference,
            #             'overhead': T_overhead,
            #             'total': T_total
            #         }
            #         with open(json_file_loc_latency, 'w') as f:
            #             json.dump(latency, f)
            #         logger.success(f"Latency for {name} calculated and saved.")
                    
            #         total_samples += 1
                    
            # total_time += time.time() - start_time
            # logger.success(f"Fold {fold} completed.")
            
    
    # logger.success(f"Total time: {total_time:.4f} sec")
    # logger.success(f"    - export time: {total_export_time:.4f} sec")
    # logger.success(f"    - process time: {total_process_time:.4f} sec")
    # logger.success(f"    - inference time: {total_inference_time:.4f} sec\n")
    
    # logger.success(f"Average latency: {total_time / total_samples:.4f} sec\n")
    
    # logger.success(f"Throughput: {total_samples / total_time:.2f} samples/sec")
    # logger.success(f"    - export time: {total_samples / total_export_time:.4f} samples/sec")
    # logger.success(f"    - process time: {total_samples / total_process_time:.4f} samples/sec")
    # logger.success(f"    - inference time: {total_samples / total_inference_time:.4f} samples/sec\n")
    
    # logger.success(f"Overhead: {total_overhead_time / total_time:.2f} sec")

    # Call plotting functions for the averages
    # plot_loss(average_train_loss, epochs, lr, train_batch_size, label_type)
    # plot_accuracy(average_eval_accuracy, epochs, lr, eval_batch_size, label_type)

    logger.success("5-Fold Cross Validation completed")
    
if __name__ == "__main__":
    set_seed(0)
    # CV_run()
    
    # label_type = "model_type"
    # fold = 1
    # with open("Naming_anomaly_detection/data_files/json_files/temp_selected_peatmoss_repos.json", "r") as f:
    #     names = json.load(f)
    # run_all_models(names, label_type, fold)
    '''
    testing individual model
    '''
    # # repo = "kazzand/ru-longformer-tiny-16384"
    # repo = "Junmai/klue-roberta-large-copa-finetuned-v1"
    # repo = "eslamxm/mbart-finetune-ar-xlsum "
    # repo = "facebook/mask2former-swin-tiny-cityscapes-semantic"
    # repo = "manishiitg/longformer-recruit-qa"
    # repo = "Classroom-workshop/assignment1-francesco"
    # repo = "hf-internal-testing/tiny-random-VisionTextDualEncoderModel-vit-bert"
    # repo = "microsoft/speecht5_tts"
    # repo = "Apocalypse-19/speecht5_finetuned_french"
    # repo = "nlp-waseda/roberta-large-japanese-with-auto-jumanpp"
    # repo = "arham061/speecht5_finetuned_voxpopuli_nl"
    # repo = "reciprocate/gpt2-tiny"
    # repo = "hf-internal-testing/tiny-random-BertLMHeadModel"
    repo = "tanaya-b/urdu_sms_hingbert"
    # repo = "hf-internal-testing/tiny-random-resnet"
    label_type = "model_type"
    fold = 1
    subprocess.Popen(["python", "Naming_anomaly_detection/DARA/latency_by_one_temp.py", "--repo", repo, "--label_type", label_type, "--fold", str(fold)])    
