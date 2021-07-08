import os

import torch
from sklearn import metrics
from torch import nn, optim


class Trainer:
    """
    A class for generalizing training process to parts of model.

    Parameters
    ----------
    model : A HuggingFace or PyTorch model
        Needs to fit for a question-answering task.
    tokenizer : A HuggingFace Fast tokenizer
        For tokenizing model or submodel inputs.
    dataloader : A dataloader.dataloaders (function) loader
        For getting train datasets ready for batch training.
    child_module_to_train : string
        A submodule of the model that is an attribute of the model.
    validation_dataloader : A dataloader.dataloaders (function) loader
        For getting validation datasets ready for batch training.
    criterion : A loss function from `torch.nn`
        For calculating loss of model or submodule.
    optimizer : A `torch.optim` optimizer
        For optimizing the model or submodule.
    val_metrics : dict of {name: sklearn metric function}
        For calcuating validation metrics.
    epochs : int
        Number of epochs to train for.
    lr : float
        Learning rate to apply in optimizer.
    batch_size : int
        The number of samples per batch.
    device : A `torch.device` object or a string representing such an object
        For handling internal tokenization.
    """

    def __init__(self,
                 model,
                 tokenizer,
                 dataloader,
                 submodule_to_train=None,
                 validation_dataloader=None,
                 criterion=nn.CrossEntropyLoss(),
                 optimizer=optim.AdamW,
                 val_metrics=['accuracy'],
                 epochs=3,
                 lr=2e-5,
                 batch_size=4,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.model = model.to(device)
        self.module_to_train = getattr(model, submodule_to_train) if submodule_to_train else model
        if self.module_to_train is not None:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            self.module_to_train.train()
            for param in self.module_to_train.parameters():
                param.requires_grad = True
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.validation_dataloader = validation_dataloader
        self.criterion = criterion
        self.optimizer = optimizer(model.parameters(), lr=lr)
        self.epochs = epochs
        self.metrics = val_metrics
        self.batch_size = batch_size
        self.device = device
        self._current_epoch = 1
        self._metric_functions = {
            'accuracy': metrics.accuracy_score,
            'f1': metrics.f1_score,
            'precision': metrics.precision_score,
            'recall': metrics.recall_score
        }

    def _run_validation(self):
        self.module_to_train.eval()
        running_loss = 0.
        y_true, y_pred = [], []
        for b, batch in enumerate(self.validation_dataloader):
            fast_kwargs = dict(return_offsets_mapping=True) if self.tokenizer.is_fast else {}
            questions, contexts, targets = batch
            # questions = batch.pop('question', None)
            # contexts = batch.pop('context', None)
            # texts = batch.pop('text', None)
            # targets = batch.pop('target')
            max_str_length = len(questions[0]) + len(contexts[0])
            inputs = list(zip(questions, contexts))
            tokenized = self.tokenizer(inputs,
                                       max_length=512,
                                       padding=True,
                                       return_tensors='pt',
                                       # return_overflowing_tokens=True,
                                       # stride=10 if max_str_length > 45 else 0,  # TODO: make this more robust
                                       ).to(self.device)
            with torch.no_grad():
                outputs = self.module_to_train(**tokenized)
                logits = outputs.get('logits',
                                     outputs.get('last_hidden_state',
                                                 torch.stack([outputs.start_logits, outputs.end_logits], dim=-1)))
                loss = self.criterion(logits, targets.to(self.device))
                y_pred.extend(torch.argmax(logits, dim=-2).flatten().cpu().detach().tolist())
                y_true.extend(torch.tensor(targets).flatten().tolist())
            running_loss += loss.item()
        y_true, y_pred = torch.tensor(y_true).numpy(), torch.tensor(y_pred).numpy()
        metric_scores = {metric: self._metric_functions.get(metric, metric)(y_true, y_pred) for metric in self.metrics}
        print(f'VALIDATION Epoch: {self._current_epoch} Batch: {b + 1} '
              f'Running Context Loss: {running_loss / (b + 1)} '
              f'Context Loss: {loss.item()} Metrics: {metric_scores}')
        return dict(loss=running_loss / (b + 1), **metric_scores)

    def _train_epoch(self):
        self.module_to_train.train()
        running_loss = 0.
        for b, batch in enumerate(self.dataloader):
            fast_kwargs = dict(return_offsets_mapping=True) if self.tokenizer.is_fast else {}
            questions, contexts, targets = batch
            # questions = batch.pop('question', None)
            # contexts = batch.pop('context', None)
            # texts = batch.pop('text', None)
            # targets = batch.pop('target')
            max_str_length = len(questions[0]) + len(contexts[0])
            inputs = list(zip(questions, contexts))
            tokenized = self.tokenizer(inputs,
                                       max_length=512,
                                       padding=True,
                                       return_tensors='pt',
                                       # return_overflowing_tokens=True,
                                       # stride=10 if max_str_length > 45 else 0,  # TODO: make this more robust
                                       ).to(self.device)
            targets = targets.to(self.device)  # TODO: adjust if question-contexts converted to multiple tensors
            outputs = self.module_to_train(**tokenized)
            logits = outputs.get('logits',
                                 outputs.get('last_hidden_state',
                                             torch.stack([outputs.start_logits, outputs.end_logits], dim=-1)))
            loss = self.criterion(logits, targets)
            running_loss += loss.item()

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            print(f'TRAIN Epoch: {self._current_epoch} Batch: {b + 1} '
                  f'Running Context Loss: {running_loss / (b + 1)} '
                  f'Context Loss: {loss.item()}')

    def train(self):
        for epoch in range(self.epochs):
            self._train_epoch()
            validation_results = self._run_validation()
            filename_args = [f'valid_{k}={v:.4f}' for k, v in sorted(validation_results.items())]
            torch.save(self.model.to('cpu').state_dict(),
                       os.path.join(
                           'modeldata',
                           f'model_{self._current_epoch}_' + '_'.join(filename_args) + '.pt'
                       ))
            self.model = self.model.to(self.device)
            self._current_epoch += 1
