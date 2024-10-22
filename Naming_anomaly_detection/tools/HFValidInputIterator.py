"""
This file contains dummy input generation for Hugging Face models.
"""
from loguru import logger
import torch
import pandas as pd
import numpy as np
from transformers import AutoModel
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tools.HFAutoClassIterator import HFAutoClassIterator
import json

import itertools

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
        inputs = {} #for models that require multiple inputs
        input_types = [torch.float16, torch.float32]
        for valid_autoclass_obj in self.valid_autoclass_obj_list:
            trial_func_list = self.func_storage.auto_get_func(valid_autoclass_obj)
            # logger.info(f"Using {valid_autoclass_obj.__class__.__name__}")
            for trial_func in trial_func_list:
                # iter_bar.update(1)
                logger.info(f"-> Trying Func {trial_func.__name__}")
                try:
                    trial_input = trial_func(valid_autoclass_obj)
                    inputs.update(trial_input)
                except Exception as emsg: # pylint: disable=broad-except
                    if valid_autoclass_obj.__class__.__name__ not in err_report:
                        err_report[valid_autoclass_obj.__class__.__name__] = {}
                    err_report[valid_autoclass_obj.__class__.__name__][trial_func.__name__]\
                        = ("CannotObtainInput", emsg)
                    # logger.warning(f"Cannot obtain input for {self.hf_repo_name} due to {emsg}")
                    continue
                # if "voice" in trial_func.__name__ and "Wav2Vec2" not in valid_autoclass_obj.__class__.__name__:    
                #     trial_input["input_features"] = trial_input['input_features'].to(torch.float32)
                #     trial_input["decoder_input_ids"] = torch.tensor([[self.model.config.decoder_start_token_id]], dtype=torch.long)
                # logger.success(trial_input)
                try:
                    self.model(**trial_input)
                    logger.success(f"Find an input for {self.hf_repo_name}")
                    return trial_input
                except Exception as emsg: # pylint: disable=broad-except
                    if valid_autoclass_obj.__class__.__name__ not in err_report:
                        err_report[valid_autoclass_obj.__class__.__name__] = {}
                    err_report[valid_autoclass_obj.__class__.__name__][trial_func.__name__]\
                        = ("InferenceError", emsg)
                
                # logger.warning(f"Cannot infer {self.hf_repo_name} due to {emsg}")
                if len(inputs) >= 2:
                    combinations = list(itertools.combinations(inputs.items(), 2))
                    for (input1, input2) in combinations:
                        # logger.info(f"{input1}, {input2}\n")
                        for torch_type in input_types:    
                            input_dict = {input1[0]: input1[1], input2[0]: input2[1]} # fix to try all combinations  #input1[0]: input1[1].to(torch.long)
                            # logger.success(f"{input_dict}, {input1[1].dtype}, {input2[1].dtype}")
                            try:
                                self.model(**input_dict)
                                # logger.success(f"Find an input for {self.hf_repo_name}, {input_dict.keys()}")
                                return input_dict
                            except RuntimeError as emsg: 
                                if "Input type (float) and bias type (c10::Half) should be the same" in str(emsg):
                                    continue
                                if "Input type (c10::Half) and bias type (float) should be the same" in str(emsg):
                                    continue
                                if str([input1[0], input2[0]]) not in err_report:
                                    err_report[str([input1[0], input2[0]])] = {}
                                err_report[str([input1[0], input2[0]])][trial_func.__name__]\
                                    = ("InferenceError", str(emsg))
                            except Exception as emsg: # pylint: disable=broad-except
                                if str([input1[0], input2[0]]) not in err_report:
                                    err_report[str([input1[0], input2[0]])] = {}
                                err_report[str([input1[0], input2[0]])][trial_func.__name__]\
                                    = ("InferenceError", emsg)
                                # logger.warning(f"Cannot infer {self.hf_repo_name} [{input1[0]}, {input2[0]}] due to {emsg}")

        logger.error(f"Cannot find a valid input for {self.hf_repo_name} or Request Time Out")
        # exit()
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
        # torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        # self.img_1_3_244_244 = torch.rand(1, 3, 224, 224)

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
    # def t_txt_10_tts(self, auto_class_obj):
    #     return auto_class_obj(text="T", return_tensors="pt")['input_ids']
    def t_txt_10_cuda(self, auto_class_obj):
        return auto_class_obj("Test Input", return_tensors="pt").to('cuda')
    def t_txt_10_labels(self, auto_class_obj):  #VisionEncoderDecoder
        encoded_input = auto_class_obj("Test Input", return_tensors="pt")
        return {"labels": encoded_input['input_ids']}
    def t_txt_input_ids(self, auto_class_obj):
        encoded_input = auto_class_obj("Test Input", return_tensors="pt")
        return {"input_ids": encoded_input['input_ids']}
    def t_enc_txt_10(self, auto_class_obj):
        return auto_class_obj.encode("Test Input", return_tensors="pt")
    def t_enc_txt_10_cuda(self, auto_class_obj):
        return auto_class_obj("Test Input", return_tensors="pt").input_ids.to('cuda')
    def t_enc_dec_t5_10(self, auto_class_obj):
        encoded_input = auto_class_obj("Test Input", return_tensors="pt")
        decoder_input_ids = encoded_input['input_ids'].clone()
        decoder_input_ids[:] = auto_class_obj.pad_token_id
        return {"input_ids": encoded_input['input_ids'], "decoder_input_ids": decoder_input_ids}
    def t_txt_decoder_input_ids(self, auto_class_obj):
        encoded_input = auto_class_obj("Test Input", return_tensors="pt")
        return {"decoder_input_ids": encoded_input['input_ids']}
    def t_img_1_3_224_224_pixel_values(self, auto_class_obj):
        encoded_input = auto_class_obj(images=torch.rand(1, 3, 224, 224), do_rescale=False, return_tensors="pt")
        return {"pixel_values": encoded_input['pixel_values']}
    
    #IndexError: index out of range in self
    def t_z_clamp_input_ids(self, auto_class_obj):
        encoded_input = auto_class_obj("Test Input", return_tensors="pt")
        max_val = auto_class_obj.pad_token_id
        input_ids = torch.clamp(encoded_input["input_ids"], min=0, max=max_val)
        encoded_input["input_ids"] = input_ids
        return encoded_input
         
    # TODO: add tokenizer for voice models

    # AutoFeatureExtractor
    def fe_voice_dummy(self, auto_class_obj):
        with open("./tools/librispeech_asr_dummy.json", "r") as f:
            ds = json.load(f)
        inputs = auto_class_obj(audio=ds["audio"]["array"], sampling_rate=16000, return_tensors='pt')
        inputs["labels"] = auto_class_obj(text_target=ds['text'], return_tensors="pt").input_ids
        return inputs
    def fe_voice_dummy2(self, auto_class_obj):
        with open("./tools/librispeech_asr_dummy.json", "r") as f:
            ds = json.load(f)
        return auto_class_obj(audio=ds["audio"]["array"], sampling_rate=16000, return_tensors='pt')
    def fe_img_3_224_224(self, auto_class_obj):
        return auto_class_obj(images=torch.rand(3, 224, 224), do_rescale=False, return_tensors="pt")
    def fe_img_1_3_224_224(self, auto_class_obj):
        return auto_class_obj(images=torch.rand(1, 3, 224, 224), do_rescale=False, return_tensors="pt")
    def fe_img_1_3_224_224_pixel(self, auto_class_obj):
        return auto_class_obj(images=torch.rand(1, 3, 224, 224), do_rescale=False, return_tensors="pt")["pixel_values"]
    def fe_voice_sr8k(self, auto_class_obj):
        return auto_class_obj(np.random.randn(8000), sampling_rate=8000, return_tensors='pt')
    def fe_voice_sr16k(self, auto_class_obj):
        return auto_class_obj(np.random.randn(16000), sampling_rate=16000, return_tensors='pt') #You have to specify either decoder_input_ids or decoder_inputs_embeds
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
        return auto_class_obj(images=torch.rand(1, 3, 224, 224), do_rescale=False, return_tensors="pt")
    def ip_img_1_3_224_224_pixel(self, auto_class_obj):
        encoded_input = auto_class_obj(images=torch.rand(3, 224, 224), do_rescale=False, return_tensors="pt")
        return {"pixel_values": encoded_input['pixel_values']}
    def ip_img_1_3_224_224_pixel_values(self, auto_class_obj):
        return auto_class_obj(images=torch.rand(1, 3, 224, 224), do_rescale=False, return_tensors="pt")["pixel_values"]
    
    # AutoProcessor

    def p_voice_dummy(self, auto_class_obj):
        with open("./tools/librispeech_asr_dummy.json", "r") as f:
            ds = json.load(f)
        return auto_class_obj(audio=ds["audio"]["array"], sampling_rate=16000, return_tensors='pt')
    def p_voice_dummy_t5(self, auto_class_obj):
        with open("./tools/librispeech_asr_dummy.json", "r") as f:
            ds = json.load(f)
        inputs = auto_class_obj(audio=ds["audio"]["array"], sampling_rate=16000, return_tensors='pt')
        inputs["labels"] = auto_class_obj(text_target=ds["text"], return_tensors="pt").input_ids
        return inputs
    def p_txt_10(self, auto_class_obj):
        return auto_class_obj("Test Input", return_tensors="pt")
    def p_txt_10_txt(self, auto_class_obj):
        return auto_class_obj(text="Test Input", return_tensors="pt")
    def p_img_1_3_224_224(self, auto_class_obj):
        return auto_class_obj(images=torch.rand(1, 3, 224, 224), return_tensors="pt")
    def p_pd_df(self, auto_class_obj):
        return auto_class_obj(pd.DataFrame(), return_tensors='pt')
    def p_voice_sr8k(self, auto_class_obj):
        return auto_class_obj(np.random.randn(8000), sampling_rate=8000, return_tensors='pt')
    def p_voice_sr16k(self, auto_class_obj):
        return auto_class_obj(torch.randn(16000, dtype=torch.float16).numpy(), sampling_rate=16000, return_tensors='pt')
    def p_voice_sr44k(self, auto_class_obj):
        return auto_class_obj(np.random.randn(44100), sampling_rate=44100, return_tensors='pt')
    def p_voice_sr48k(self, auto_class_obj):
        return auto_class_obj(np.random.randn(48000), sampling_rate=48000, return_tensors='pt')
    def p_voice_sr96k(self, auto_class_obj):
        return auto_class_obj(np.random.randn(96000), sampling_rate=96000, return_tensors='pt')
    def p_voice_sr192k(self, auto_class_obj):
        return auto_class_obj(np.random.randn(192000), sampling_rate=192000, return_tensors='pt')
    def p_vision_encoder_decoder(self, auto_class_obj):
        pixel_values = auto_class_obj(torch.rand(1, 3, 224, 224), return_tensors="pt").pixel_values
        labels = auto_class_obj.tokenizer("Test Input", return_tensors="pt").input_ids
        return {"pixel_values": pixel_values, "labels": labels}
    
if __name__ == "__main__":
    # repo_name = "anton-l/distilhubert-ft-keyword-spotting"
    repo_name = "microsoft/resnet-50"
    model = AutoModel.from_pretrained(repo_name)
    in_iter = HFValidInputIterator(model, repo_name, None)
    print(in_iter.get_valid_input())
