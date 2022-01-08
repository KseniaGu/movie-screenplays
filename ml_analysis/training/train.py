import os
import random

import mlflow
from transformers import AdamW, get_cosine_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from sklearn.metrics import balanced_accuracy_score, f1_score
import torch
import numpy as np
from tqdm import tqdm
from time import time
from shutil import copytree

from ml_analysis.ml_utils import format_time, plot_conf_matr


class Trainer:
    def __init__(self, config, model, train_dataloader, validation_dataloader):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.crit = CrossEntropyLoss()
        self._create_optimizer()
        self._create_scheduler()

        self.model.to(self.config['train']['device'])

        self.global_step = 0

    def set_training_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def log_training_params(self, optimizer_config):
        for key, val in optimizer_config.items():
            mlflow.log_param(key, val)
        task = self.config['train']['task']
        nrof_classes = len(self.config['train']['classes_names'][task])
        mlflow.log_param('nrof_classes', nrof_classes)

    def log_metrics(self, metrics_dict, elapsed_time):
        training_status_msg = []

        for metric, val in metrics_dict.items():
            mlflow.log_metric(metric, val, step=self.global_step)
            training_status_msg.append(metric + ': ' + str(val))

        print(f'Elapsed time: {elapsed_time}\n' + '\n'.join(training_status_msg))

    def log_additional_data(self, file_name):
        mlflow.log_artifact(file_name)

    def _create_optimizer(self):
        optimizer = self.config['train']['optimizer']
        optimizer_config = self.config['train']['optimizer_params'][optimizer]
        self.log_training_params(optimizer_config)

        if optimizer == AdamW.__name__:
            self.optimizer = AdamW(self.model.parameters(),
                                   lr=optimizer_config['lr'],
                                   eps=optimizer_config['eps'],
                                   weight_decay=optimizer_config['weight_decay'])
        else:
            raise Exception('Unknown optimizer')

    def _create_scheduler(self):
        if self.optimizer is not None:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.config['train']['nrof_warmup_steps'],
                num_training_steps=self.config['train']['nrof_steps_for_shed']
            )
        else:
            raise Exception('Unknown optimizer')

    def save_model(self, step=None):
        if not os.path.exists(self.config['paths']['ckpt_dir']):
            os.makedirs(self.config['paths']['ckpt_dir'])
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                'global_step': self.global_step
            },
            os.path.join(
                self.config['paths']['ckpt_dir'],
                self.config['train']['task'] + '_' +
                self.config['train']['exp_name'] + '_' +
                'checkpoint' + ('_' + str(step) if step else '')))
        print('Model saved...')

    def load_model(self):
        if self.config['train']['use_intermediate_weights']:
            ckpt = torch.load(self.config['paths']['ckpt_to_load'],
                              map_location=self.config['train']['device'])
            model_st_dict = ckpt["model"]
            model_st_dict = dict(
                [(key.replace('bert', 'transformer_model'), value) for key, value in model_st_dict.items()])
            self.model.load_state_dict(model_st_dict, strict=False)
        else:
            ckpt = torch.load(self.config['paths']['ckpt_to_load'],
                              map_location=self.config['train']['device'])
            model_st_dict = ckpt["model"]
            self.model.load_state_dict(model_st_dict)
            self.global_step = ckpt["global_step"] + 1
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
        print("Model loaded...")

    def training_step(self, batch, cur_loss, tr_losses, predicts, trues, nrof_steps, nrof_samples, start_time):
        b_labels = batch['labels'].to(self.config['train']['device'])

        self.model.zero_grad()
        output = self.model(batch)

        loss = self.crit(output, b_labels)
        cur_loss += loss.item()
        tr_losses.append(loss.item())

        _, predicted = torch.max(output, -1)
        predicts.extend(predicted.cpu().detach().numpy().flatten().tolist())
        trues.extend(b_labels.cpu().detach().numpy().flatten().tolist())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        nrof_steps += 1
        nrof_samples += len(b_labels)

        if self.global_step % self.config['train']['train_logging_step'] == 0:
            metric_eval_examples_num = self.config['train']['metric_eval_examples_num']
            tr_losses = tr_losses[-metric_eval_examples_num:]
            trues = trues[-metric_eval_examples_num:]
            predicts = predicts[-metric_eval_examples_num:]

            bal_tr_accuracy = balanced_accuracy_score(trues, predicts)
            # tr_f1_score = f1_score(trues, predicts)

            metrics_dict = {
                'train_loss': cur_loss / nrof_steps,
                'train_average_loss': np.mean(tr_losses),
                'train_bal_accuracy': bal_tr_accuracy,
                'learning_rate': self.optimizer.state_dict()["param_groups"][0]["lr"]
            }
            self.log_metrics(metrics_dict, format_time(time() - start_time))
            print('Results examples:\nTrues: {}\nPredicts:{}'.format(trues, predicts))

            cur_loss, nrof_steps = 0., 0

        return cur_loss, tr_losses, predicts, trues, nrof_steps, nrof_samples

    def train(self, train_dataloader=None, seed=None):
        train_dataloader = train_dataloader \
            if not train_dataloader is None else self.train_dataloader

        if seed is not None:
            self.set_training_seed(seed)
        else:
            self.set_training_seed(self.config['train']['seed'])

        if self.config['train']['load_model']:
            self.load_model()

        self.model.train()
        start_time = time()

        tr_losses = []
        cur_loss, nrof_steps, = 0., 0,

        nrof_samples_current, nrof_samples = 0, 0
        train_dataloader_len = len(train_dataloader)
        print(f'Current global step: {self.global_step}')
        print(f'Train loader length: {train_dataloader_len}')

        for epoch in tqdm(range(self.config['train']['nrof_epochs'])):
            predicts, trues = [], []
            for step, batch in enumerate(train_dataloader):
                if epoch * train_dataloader_len + step < self.global_step:
                    continue
                print(f'Epoch: {epoch}, step: {step}')

                cur_loss, tr_losses, predicts, trues, nrof_steps, nrof_samples = self.training_step(batch,
                                                                                                    cur_loss,
                                                                                                    tr_losses,
                                                                                                    predicts,
                                                                                                    trues,
                                                                                                    nrof_steps,
                                                                                                    nrof_samples,
                                                                                                    start_time)

                if self.global_step % self.config['train']['global_ckpt_saving_step'] == 0:
                    self.save_model()
                    # colab setup:
                    '''
                    try:
                        copytree(
                            os.path.join(self.config['paths']['logs_dir'], 'mlruns'),
                            '_'.join((self.config['paths']['mlruns'], 
                                      self.config['train']['exp_name'], 
                                      str(self.global_step)))
                        )
                    except Exception as e:
                        print(e)
                        pass
                    '''
                if self.global_step % self.config['train']['running_ckpt_saving_step'] == 0:
                    self.save_model(step=self.global_step)

                if self.config['train']['to_validate'] \
                        and self.global_step % self.config['train']['validation_step'] == 0:
                    self.validate()

            self.save_model(step=self.global_step)

            # colab setup:
            '''
            try:
                copytree(
                    os.path.join(self.config['paths']['logs_dir'], 'mlruns'),
                    '_'.join((self.config['paths']['mlruns'], 
                              self.config['train']['exp_name'], 
                              str(self.global_step)))
                )
            except Exception as e:
                print(e)
                pass
            '''

    def validate(self, validation_dataloader=None, to_load=False):
        print('Validation...')
        if to_load:
            self.load_model()
        validation_dataloader = validation_dataloader \
            if not validation_dataloader is None else self.validation_dataloader
        start_time = time()
        self.model.eval()

        probs, predicts, trues = [], [], []
        nrof_steps, val_loss, nrof_cor_predicts, nrof_samples = 0, 0., 0, 0

        for i, batch in tqdm(enumerate(validation_dataloader)):
            with torch.no_grad():
                b_labels = batch['labels'].to(self.config['train']['device'])

                output = self.model(batch)
                val_loss += self.crit(output, b_labels).item()

                sig_probs, predicted = torch.max(output.sigmoid(), -1)
                probs.extend(sig_probs.cpu().detach().numpy().flatten().tolist())
                predicts.extend(predicted.cpu().detach().numpy().flatten().tolist())
                trues.extend(b_labels.cpu().detach().numpy().flatten().tolist())

                nrof_steps += 1
                nrof_samples += len(b_labels)

        avg_val_loss = val_loss / nrof_steps
        validation_time = (time() - start_time)

        avg_val_accuracy = balanced_accuracy_score(trues, predicts)

        val_metrics_dict = {
            "val_loss": avg_val_loss,
            "val_accuracy": avg_val_accuracy
        }

        self.log_metrics(val_metrics_dict, time() - start_time)

        classes_names = self.config['train']['classes_names'][self.config['train']['task_name']]
        plot_conf_matr(trues, predicts, title=str(self.global_step) + '_',
                       nrof_classes=len(classes_names),
                       classes_names=classes_names)
        self.log_additional_data(str(self.global_step) + '_conf_matrix.png')

        print('Results examples:\nTrues: {}\nPredicts:{}'.format(trues, predicts))
        print(f'Validation done after {format_time(validation_time)}.')

        return
