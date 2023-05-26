from torch import nn
from transformers import RobertaModel


class RobertaClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(768, 10)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        roberta_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        last_hidden_state = roberta_output[0]
        cls_hidden_state = last_hidden_state[:, 0]
        cls_hidden_state = self.pre_classifier(cls_hidden_state)
        cls_hidden_state = self.dropout(cls_hidden_state)
        output = self.classifier(cls_hidden_state)
        return output
    