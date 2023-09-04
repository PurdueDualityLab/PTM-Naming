import json
from tokenizers import BertWordPieceTokenizer
from tokenizers.trainers import WordPieceTrainer

# Load your list of sentences
with open("/depot/davisjam/data/chingwo/PTM-Naming/vectorizer/pytorch/layer_name_data.json", "r") as f:
    training_corpus = json.load(f)

# Initialize a tokenizer
tokenizer = BertWordPieceTokenizer()

# Train the tokenizer
tokenizer.train(
    ["/depot/davisjam/data/chingwo/PTM-Naming/vectorizer/pytorch/layer_name_data.json"],
    vocab_size=10000,
    min_frequency=2,
    show_progress=True,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    limit_alphabet=1000,
    wordpieces_prefix="##",
)

# Save the tokenizer
tokenizer.save_model("/depot/davisjam/data/chingwo/PTM-Naming/vectorizer/pytorch")
