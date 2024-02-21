
from tools.HFAutoClassIterator import HFAutoClassIterator
from tqdm import tqdm
from loguru import logger
import torch
import pandas as pd
import numpy as np
from transformers import AutoModel

class HFValidInputIterator():

    def __init__(
            self,
            model, 
            hf_repo_name, 
            cache_dir,
            device=None,
            trust_remote_code=False
        ):
        self.model = model
        self.hf_repo_name = hf_repo_name
        self.func_storage = TrialFunctionStorage()
        self.valid_autoclass_obj_list = HFAutoClassIterator(hf_repo_name, cache_dir=cache_dir, trust_remote_code=trust_remote_code).get_valid_auto_class_objects()
        self.require_remote_code = False
        if isinstance(self.valid_autoclass_obj_list, dict):
            logger.error(f"Cannot find a valid autoclass for {self.hf_repo_name}")
            for autoclass_type, err in self.valid_autoclass_obj_list.items():
                if "trust_remote_code" in str(err):
                    self.require_remote_code = True
                logger.error(f"-> {autoclass_type}({err})")
            return None

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(self.valid_autoclass_obj_list)
    
    def get_valid_input(self):

        # count iterations
        # iter_count = 0
        # for valid_autoclass in self.valid_autoclass_obj_list:
        #     trial_func_list = self.func_storage.auto_get_func(valid_autoclass)
        #     iter_count += len(trial_func_list)
        # iter_bar = tqdm(range(iter_count))
        
        # actual tries

        err_report = {}
        for valid_autoclass_obj in self.valid_autoclass_obj_list:
            trial_func_list = self.func_storage.auto_get_func(valid_autoclass_obj)
            logger.info(f"Using {valid_autoclass_obj.__class__.__name__}")
            for trial_func in trial_func_list:
                # iter_bar.update(1)
                logger.info(f"-> Trying Func {trial_func.__name__}")
                try:
                    trial_input = trial_func(valid_autoclass_obj)
                except Exception as emsg:
                    if valid_autoclass_obj.__class__.__name__ not in err_report:
                        err_report[valid_autoclass_obj.__class__.__name__] = {}
                    err_report[valid_autoclass_obj.__class__.__name__][trial_func.__name__] = ("CannotObtainInput", emsg)
                    continue
                try:
                    # print(trial_input)
                    trial_input.to(self.device)
                    self.model.to(self.device)
                    self.model(**trial_input) # could be yield trial_input, change it to a generator maybe??
                    logger.success(f"Find an input for {self.hf_repo_name}")
                    # iter_bar.close()
                    return trial_input.to(self.device)

                except Exception as emsg:
                    if valid_autoclass_obj.__class__.__name__ not in err_report:
                        err_report[valid_autoclass_obj.__class__.__name__] = {}
                    err_report[valid_autoclass_obj.__class__.__name__][trial_func.__name__] = ("InferenceError", emsg)
                    pass
        
        logger.error(f"Cannot find a valid input for {self.hf_repo_name} or Request Time Out")
        for autoclass_type, trial_func_dict in err_report.items():
            logger.error(f"Error report for {autoclass_type}:")
            for trial_func, err in trial_func_dict.items():
                logger.error(f"-> {trial_func}({err[0]}): {err[1]}")
        # iter_bar.close()
        return None

class TrialFunctionStorage():

    def __init__(self):
        pass

    def get_autotokenizer_func(self):
        return [
            self.t_txt_10,
            self.t_enc_txt_10
        ]

    def get_autofeatureextractor_func(self):
        return [
            self.fe_img_1_3_224_224,
            self.fe_voice_sr8k,
            self.fe_voice_sr16k,
            self.fe_voice_sr44k,
            self.fe_voice_sr48k,
            self.fe_voice_sr96k,
            self.fe_voice_sr192k
        ]

    def get_autoimage_processor_func(self):
        return [
            self.ip_img_1_3_224_224
        ]

    def get_autoprocessor_func(self):
        return [
            self.p_txt_10,
            self.p_img_1_3_224_224,
            self.p_pd_df,
            self.p_voice_sr8k,
            self.p_voice_sr16k,
            self.p_voice_sr44k,
            self.p_voice_sr48k,
            self.p_voice_sr96k,
            self.p_voice_sr192k
        ]

    def auto_get_func(self, auto_class_obj): # might not be right, need to check
        if "Tokenizer" in auto_class_obj.__class__.__name__:
            return self.get_autotokenizer_func()
        elif "FeatureExtractor" in auto_class_obj.__class__.__name__:
            return self.get_autofeatureextractor_func()
        elif "ImageProcessor" in auto_class_obj.__class__.__name__:
            return self.get_autoimage_processor_func()
        elif "Processor" in auto_class_obj.__class__.__name__:
            return self.get_autoprocessor_func()
        else:
            raise ValueError("Incorrect object type.")
        
    # AutoTokenizer
    
    def t_txt_10(self, auto_class_obj):
        return auto_class_obj("Test Input", return_tensors="pt")
    def t_enc_txt_10(self, auto_class_obj):
        return auto_class_obj.encode("Test Input", return_tensors="pt")
    
    # AutoFeatureExtractor

    def fe_img_1_3_224_224(self, auto_class_obj):
        return auto_class_obj(images=torch.rand(1, 3, 224, 224), return_tensors="pt")["pixel_values"]
    def fe_voice_sr8k(self, auto_class_obj):
        return auto_class_obj(np.random.randn(8000), sampling_rate=8000, return_tensors='pt')
    def fe_voice_sr16k(self, auto_class_obj):
        return auto_class_obj(np.random.randn(16000), sampling_rate=16000, return_tensors='pt')
    def fe_voice_sr44k(self, auto_class_obj):
        return auto_class_obj(np.random.randn(44100), sampling_rate=44100, return_tensors='pt')
    def fe_voice_sr48k(self, auto_class_obj):
        return auto_class_obj(np.random.randn(48000), sampling_rate=48000, return_tensors='pt')
    def fe_voice_sr96k(self, auto_class_obj):
        return auto_class_obj(np.random.randn(96000), sampling_rate=96000, return_tensors='pt')
    def fe_voice_sr192k(self, auto_class_obj):
        return auto_class_obj(np.random.randn(192000), sampling_rate=192000, return_tensors='pt')
    
    # AutoImageProcessor

    def ip_img_1_3_224_224(self, auto_class_obj):
        return auto_class_obj(images=torch.rand(1, 3, 224, 224), return_tensors="pt")
    
    # AutoProcessor

    def p_txt_10(self, auto_class_obj):
        return auto_class_obj("Test Input", return_tensors="pt")
    def p_img_1_3_224_224(self, auto_class_obj):
        return auto_class_obj(images=torch.rand(1, 3, 224, 224), return_tensors="pt")
    def p_pd_df(self, auto_class_obj):
        return auto_class_obj(pd.DataFrame(), return_tensors='pt')
    def p_voice_sr8k(self, auto_class_obj):
        return auto_class_obj(np.random.randn(8000), sampling_rate=8000, return_tensors='pt')
    def p_voice_sr16k(self, auto_class_obj):
        return auto_class_obj(np.random.randn(16000), sampling_rate=16000, return_tensors='pt')
    def p_voice_sr44k(self, auto_class_obj):
        return auto_class_obj(np.random.randn(44100), sampling_rate=44100, return_tensors='pt')
    def p_voice_sr48k(self, auto_class_obj):
        return auto_class_obj(np.random.randn(48000), sampling_rate=48000, return_tensors='pt')
    def p_voice_sr96k(self, auto_class_obj):
        return auto_class_obj(np.random.randn(96000), sampling_rate=96000, return_tensors='pt')
    def p_voice_sr192k(self, auto_class_obj):
        return auto_class_obj(np.random.randn(192000), sampling_rate=192000, return_tensors='pt')
    
if __name__ == "__main__":
    repo_name = "anton-l/distilhubert-ft-keyword-spotting"
    model = AutoModel.from_pretrained(repo_name)
    in_iter = HFValidInputIterator(model, repo_name, None)
    print(in_iter.get_valid_input())