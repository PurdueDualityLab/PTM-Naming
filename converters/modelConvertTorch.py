# To be used under general env
# module use /depot/davisjam/data/chingwo/general_env/modules
# module load conda-env/general_env-py3.8.5

import torch

def torch_to_onnx(
        torch_model,
        directory,
        dummy_input=torch.randn(1, 3, 224, 224), 
        input_name=['input'],
        output_name=['output']
        ):
    torch.onnx.export(
        torch_model, 
        dummy_input, 
        directory, 
        verbose=False, 
        input_names=input_name, 
        output_names=output_name, 
        do_constant_folding=False, 
        opset_version=16,
        training=torch.onnx.TrainingMode.TRAINING
    )


