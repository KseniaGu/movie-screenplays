import torch
import torch.nn as nn
from torch.nn import LSTM
import torch.nn.functional as F


class ScreenplayAssessor(nn.Module):
    """BERT-based siamese model for assessing long screenplay documents."""
    def __init__(self, transformer_model, config):
        super().__init__()
        self.config = config
        self.transformer_model = transformer_model
        for param in self.transformer_model.parameters():
            param.requires_grad = True

        task = self.config['train']['task']
        self.dropout = nn.Dropout(self.config['train']['dropout'])
        self.cls_ff = nn.Linear(self.config['train']['embedding_size'], len(self.config['train']['classes_names'][task]))
        self.lstm = LSTM(self.config['train']['embedding_size'], self.config['train']['embedding_size'])

    def forward(self, batch):
        input_ids = batch['inputs'].to(self.config['train']['device'])
        input_masks = batch['attention_masks'].to(self.config['train']['device'])

        bert_output = torch.zeros(self.config['train']['batch_size']['train'],
                                  self.config['tokenization']['screenplay_dataset']['max_scene_number'],
                                  self.config['train']['embedding_size'],
                                  device=self.config['train']['device'])

        for i in range(input_ids.shape[1]):
            bert_output[:, i, :] = self.transformer_model(
                input_ids[:, i, :], attention_mask=input_masks[:, i, :]).last_hidden_state[:, 0, :]

        script_vector = torch.mean(bert_output, 1)
        ff_res = self.cls_ff(self.dropout(F.relu(script_vector)))

        return ff_res
