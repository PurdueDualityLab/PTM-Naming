"""
This file contains dummy input generation for Hugging Face models.
"""

from loguru import logger
import torch
import pandas as pd
import numpy as np
from transformers import AutoModel
from tools.HFAutoClassIterator import HFAutoClassIterator

class HFValidInputIterator():
    """
    This class is used to generate valid inputs for Hugging Face models.

    Attributes:
        model: The Hugging Face model.
        hf_repo_name: The Hugging Face repository name.
        func_storage: The storage for trial functions.
        valid_autoclass_obj_list: The list of valid autoclass objects.
        device: The device to use for inference.
    """
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
        self.valid_autoclass_obj_list = \
            HFAutoClassIterator(
                hf_repo_name,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code
            ).get_valid_auto_class_objects()
        self.err_type = ""
        if isinstance(self.valid_autoclass_obj_list, dict):
            logger.error(f"Cannot find a valid autoclass for {self.hf_repo_name}")
            for autoclass_type, err in self.valid_autoclass_obj_list.items():
                if "trust_remote_code" in str(err):
                    self.err_type = "requires_remote_code"
                elif "does not sppear to have a file named preprocessor_config.json" in str(err):
                    self.err_type = "no_preprocessor_config"
                logger.error(f"-> {autoclass_type}({err})")
            return None

        self.device = device if device is not None \
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_valid_input(self):
        """
        This function generates a valid input for the Hugging Face model.

        Returns:
            A valid input for the Hugging Face model.
        """
        err_report = {}
        for valid_autoclass_obj in self.valid_autoclass_obj_list:
            trial_func_list = self.func_storage.auto_get_func(valid_autoclass_obj)
            logger.info(f"Using {valid_autoclass_obj.__class__.__name__}")
            for trial_func in trial_func_list:
                # iter_bar.update(1)
                logger.info(f"-> Trying Func {trial_func.__name__}")
                try:
                    trial_input = trial_func(valid_autoclass_obj)
                except Exception as emsg: # pylint: disable=broad-except
                    if valid_autoclass_obj.__class__.__name__ not in err_report:
                        err_report[valid_autoclass_obj.__class__.__name__] = {}
                    err_report[valid_autoclass_obj.__class__.__name__][trial_func.__name__]\
                        = ("CannotObtainInput", emsg)
                    continue
                try:
                    self.model(**trial_input)
                    logger.success(f"Find an input for {self.hf_repo_name}")
                    return trial_input
                except Exception as emsg: # pylint: disable=broad-except
                    if valid_autoclass_obj.__class__.__name__ not in err_report:
                        err_report[valid_autoclass_obj.__class__.__name__] = {}
                    err_report[valid_autoclass_obj.__class__.__name__][trial_func.__name__]\
                        = ("InferenceError", emsg)

        logger.error(f"Cannot find a valid input for {self.hf_repo_name} or Request Time Out")
        for autoclass_type, trial_func_dict in err_report.items():
            logger.error(f"Error report for {autoclass_type}:")
            for trial_func, err in trial_func_dict.items():
                logger.error(f"-> {trial_func}({err[0]}): {err[1]}")
        return (err_report, "ErrMark")

class TrialFunctionStorage():
    """
    This class is used to store trial functions for Hugging Face models.

    Attributes:
        None
    """
    def __init__(self):
        pass

    def auto_get_func(self, auto_class_obj):
        """
        This function retrieves the trial functions for the Hugging Face model.

        Args:
            auto_class_obj: The Hugging Face model.

        Returns:
            The list of trial functions for the Hugging Face model.
        """
        prefix_map = {
            "Tokenizer": "t_",
            "FeatureExtractor": "fe_",
            "ImageProcessor": "ip_",
            "Processor": "p_",
        }
        class_name = auto_class_obj.__class__.__name__
        prefix = None
        for key, value in prefix_map.items():
            if key in class_name:
                prefix = value
                break

        if prefix is None:
            raise ValueError("Incorrect object type.")

        # Dynamically retrieve methods with the matching prefix
        return [getattr(self, method_name) for method_name in dir(self)\
            if callable(getattr(self, method_name)) \
            and method_name.startswith(prefix)]


    # Trial Functions List

    # AutoTokenizer

    def t_txt_10(self, auto_class_obj):
        return auto_class_obj("Test Input", return_tensors="pt")
    def t_enc_txt_10(self, auto_class_obj):
        return auto_class_obj.encode("Test Input", return_tensors="pt")
    def t_enc_dec_t5_10(self, auto_class_obj):
        encoded_input = auto_class_obj("Test Input", return_tensors="pt")
        decoder_input_ids = encoded_input['input_ids'].clone()
        decoder_input_ids[:] = auto_class_obj.pad_token_id
        return {"input_ids": encoded_input['input_ids'], "decoder_input_ids": decoder_input_ids}

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
