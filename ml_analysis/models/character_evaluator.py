import torch.nn as nn
from transformers import BertForSequenceClassification


class CharacterEvaluator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        task = self.config['train']['task']
        nrof_classes = len(self.config['train']['classes_names'][task])
        self.model = BertForSequenceClassification.from_pretrained(self.config['train']['pretrained_model_type'],
                                                                   num_labels=nrof_classes,
                                                                   output_attentions=False,
                                                                   output_hidden_states=False)

    def forward(self, batch):
        input_ids = batch['inputs'].to(self.config['train']['device'])
        attention_masks = batch['attention_masks'].to(self.config['train']['device'])

        logits = self.model(input_ids, attention_masks, return_dict=True).logits
        return logits
