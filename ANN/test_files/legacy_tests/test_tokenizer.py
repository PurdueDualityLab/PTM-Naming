from tokenizers import BertWordPieceTokenizer

# Initialize the tokenizer and load from the saved vocab file
tokenizer = BertWordPieceTokenizer("/depot/davisjam/data/chingwo/PTM-Naming/vectorizer/pytorch/vocab.txt")

# Now you can encode text using the loaded tokenizer
encoded = tokenizer.encode("Undefined __getitem__ to __rsub__ mul add softmax Dropout matmul permute contiguous view Linear Dropout add LayerNorm Linear GELUActivation Linear Dropout add LayerNorm Linear view permute matmul permute contiguous view Linear Dropout add LayerNorm Linear GELUActivation Linear Dropout add LayerNorm Linear view permute transpose matmul div add softmax Dropout matmul permute contiguous view Linear Dropout add LayerNorm Linear GELUActivation Linear Dropout add LayerNorm Linear view permute matmul div add softmax Dropout matmul permute contiguous view Linear Dropout add LayerNorm Linear GELUActivation Linear Dropout add LayerNorm Linear view permute transpose matmul div add softmax Dropout matmul permute contiguous view Linear Dropout add LayerNorm Linear GELUActivation Linear Dropout add LayerNorm Linear")
print(encoded.ids)  # Output: token ids
print(encoded.tokens)  # Output: tokens