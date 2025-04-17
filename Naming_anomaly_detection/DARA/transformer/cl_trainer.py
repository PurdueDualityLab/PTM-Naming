from transformers import Trainer
from transformers.modeling_outputs import SequenceClassifierOutput

class CustomCLTrainer(Trainer):
    def __init__(self, model, args, train_dataset, eval_dataset, compute_metrics, loss_fn=None, **kwargs):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics, 
            **kwargs
        )
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs, output_hidden_states=True)
        
        cls_embedding = outputs.hidden_states[-1][:, 0, :]

        logits = outputs.logits
        loss = self.loss_fn(cls_embedding, logits, labels)
        outputs.hidden_states = None

        clean_outputs = SequenceClassifierOutput(logits=logits)
        return (loss, clean_outputs) if return_outputs else loss
