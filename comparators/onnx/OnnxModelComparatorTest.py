
import OnnxModelComparator
import onnx

# To be used under general env
# module use /depot/davisjam/data/chingwo/general_env/modules
# module load conda-env/general_env-py3.8.5


# Model path

MODEL_PATH = '/depot/davisjam/data/chingwo/PTM-Naming/test_models/'


# Test for operator difference

def operatorDiffTest() -> bool:
    tfmg_model = onnx.load(MODEL_PATH + "resnet101_v1_tfmg.onnx")
    onnx_model = onnx.load(MODEL_PATH + "resnet101-v1-7.onnx")
    return OnnxModelComparator.analyzeModelLayerDiff(tfmg_model, onnx_model) == "op"


# Test for dimension difference

def dimDiffTest() -> bool:
    torch_model = onnx.load(MODEL_PATH + "resnet101-v1-torch.onnx")
    onnx_model = onnx.load(MODEL_PATH + "resnet101-v1-7.onnx")
    return OnnxModelComparator.analyzeModelLayerDiff(torch_model, onnx_model) == "dim"


# Test for parameter difference (TODO)

def paramDiffTest() -> bool:
    return False


# Test for same model

def sameModelTest() -> bool:
    onnx_model = onnx.load(MODEL_PATH + "resnet101-v1-7.onnx")
    return OnnxModelComparator.analyzeModelLayerDiff(onnx_model, onnx_model) == "none"


# Perform all tests

def test() -> None:
    assert operatorDiffTest()
    assert dimDiffTest()
    # assert paramDiffTest() TODO
    assert sameModelTest()
    print("all tests passed")

if __name__ == '__main__':
    test()