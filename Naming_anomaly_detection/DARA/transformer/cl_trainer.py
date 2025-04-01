from transformers import Trainer, TrainingArguments
import torch
import torch.nn.functional as F

class ContrastiveCrossEntropyTrainer(Trainer):
    def __init__(self, model, args, train_dataset, eval_dataset, compute_metrics, loss_fn, **kwargs):
        super().__init__(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, compute_metrics=compute_metrics)#, **kwargs)
        self.loss_fn = loss_fn
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs, output_hidden_states=True)

        embeddings = outputs.hidden_states[-1][:, 0, :]  # Shape: (batch_size, hidden_size), equivalent to hidden representation of <s> / <CLS> token
        logits = outputs.logits

        loss = self.loss_fn(embeddings, logits, labels)

        return (loss, outputs) if return_outputs else loss