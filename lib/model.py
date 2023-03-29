import torch
from transformers import BertModel


class BertForRegression(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.regressor = torch.nn.Linear(config.hidden_size, 1)
        
    def forward(self, input_ids, **kwargs):
        outputs = super().forward(input_ids=input_ids, **kwargs)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        concreteness_score = self.regressor(cls_output)

        return concreteness_score