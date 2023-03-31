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


def clip_predict_word(word, tokenizer, model, device=torch.device('cpu')):
    with torch.no_grad():
        outputs = model(**tokenizer(word, padding='max_length', return_tensors="pt", max_length=10).to(device))
    return outputs.item()

def bert_predict_word(word, tokenizer, model, device=torch.device('cpu')):
    input_ids = tokenizer(word, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids)
    return outputs.logits.item()
