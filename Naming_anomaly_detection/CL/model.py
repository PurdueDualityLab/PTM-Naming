import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
        
class CLSingleHead(nn.Module):
    def __init__(self, args, model, num_label):
        super(CLSingleHead, self).__init__()
        self.model = model
        self.config = model.config
        self.hidden_size = model.config.hidden_size
        self.num_label = num_label
        self.classifier_dropout = 0.2
        # (
        #     model.config.classifier_dropout if model.config.classifier_dropout is not None else model.config.hidden_dropout_prob
        # )
        self.fc = self._build_fcn()

    def forward(self, input_ids, attention_mask): 
        input_ids = torch.squeeze(input_ids, dim=1)
        attention_mask = torch.squeeze(attention_mask, dim=1)

        output = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled_output = output[:,0,:] # equivalent to hidden representation of <s> / <CLS> token
        label_type = self.fc(pooled_output)
        
        return output, label_type

    def _build_fcn(self):
        return nn.Sequential(
            nn.Dropout(self.classifier_dropout),
            nn.Linear(self.hidden_size, self.num_label)      
        )
        
if __name__ == "__main__":
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base")
    model = CLSingleHead(model, tokenizer)
    print(model.hidden_size)
