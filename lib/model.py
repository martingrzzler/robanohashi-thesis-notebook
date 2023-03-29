import torch
from transformers import CLIPTextModelWithProjection, CLIPTextConfig

class CLIPForRegression(CLIPTextModelWithProjection):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.regressor = torch.nn.Linear(config.hidden_size, 1)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, **kwargs):
        outputs = super().forward(**kwargs)
        outputs = self.dropout(outputs.text_embeds)
        outputs = self.regressor(outputs)
        return outputs