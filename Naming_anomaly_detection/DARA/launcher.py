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
json_file_loc_flop = os.path.join(PEATMOSS_PATH, "params.json")
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
    
def run_all_models(repos):
    for repo in repos:
        if repo in params:
            print(f"{repo} already exists, skipping...")
            continue
        
        start_time = time.time()
        logger.info(f"Starting evaluation for model: {repo}")
        result = subprocess.run(["python", "Naming_anomaly_detection/DARA/latency_by_one.py", "--repo", repo])    
        end_time = time.time()
        print(f"Finished evaluation for model: {repo} in {end_time - start_time} seconds.\n", result.stdout)

    
if __name__ == "__main__":
    set_seed(0)
    with open("Naming_anomaly_detection/data_files/json_files/temp_selected_peatmoss_repos.json", "r") as f:
        repos = json.load(f)

        with torch.no_grad():
            run_all_models(repos)

    
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
    # repo = "hf-internal-testing/tiny-random-resnet"
    # repo = "sazzad-sit/whisper-small-bn-cv13-gf"
    # label_type = "model_type"
    # fold = 1
    # subprocess.Popen(["python", "Naming_anomaly_detection/DARA/latency_by_one_temp.py", "--repo", repo, "--label_type", label_type, "--fold", str(fold)])    
