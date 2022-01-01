import torch.nn as nn
from transformers import BertForNextSentencePrediction


class NextScenePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BertForNextSentencePrediction.from_pretrained(self.config['train']['pretrained_model_type'],
                                                                   num_labels=2,
                                                                   output_attentions=False,
                                                                   output_hidden_states=False)

    def forward(self, batch):
        input_ids = batch['inputs'].to(self.config['train']['device'])
        attention_masks = batch['attention_masks'].to(self.config['train']['device'])
        token_type_ids = batch['token_type_ids'].to(self.config['train']['device'])

        logits = self.model(input_ids, attention_masks, token_type_ids, return_dict=True).logits
        return logits
