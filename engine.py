import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab

from transformers import BertTokenizer

import numpy as np


class SimpleClassificator(nn.Module):
    def __init__(self, vocab_size, max_tokens):
        super(SimpleClassificator, self).__init__()
        self.max_tokens = max_tokens

        self.embedding = nn.Embedding(vocab_size, 8, padding_idx=0)
        # self.lstm = nn.LSTM(
        #     input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_lstm, dropout=1, batch_first=True
        # )
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout1d(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout1d(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout1d(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout1d(0.2),
            nn.Linear(128, 2)
        )

    def __call__(self, x: torch.FloatType):
        x = self.embedding(x)
        x = self.fc(x.reshape(len(x), -1))
        return x


class BertDataset(Dataset):
    def __init__(self, x: np.array, y: np.array, tokenizer: BertTokenizer):
        self.x = x
        self.y = y
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        text = str(self.x[i])
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=512, return_token_type_ids=False,
            padding='max_length', return_attention_mask=True, return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(self.y[i], dtype=torch.long)
        }


class SimpleDataset(Dataset):
    def __init__(self, x: np.array, y: np.array, vocab: Vocab, tensor_size: int):
        self.x = x
        self.y = y
        self.vocab = vocab
        self.tensor_size = tensor_size

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        tensor = self.vocab(self.x[i].split())
        # print(len(tensor), (self.tensor_size - len(tensor)))
        tensor += [0] * (self.tensor_size - len(tensor))  # add padding
        return torch.LongTensor(tensor), self.y[i]


def bert_train_epoch(model, optimizer, train_loader, loss_function, device='cuda'):
    losses = []
    correct_predictions = 0

    for data in train_loader:
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        targets = data["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        preds = torch.argmax(outputs.logits, dim=1)
        loss = loss_function(outputs.logits, targets)

        correct_predictions += torch.sum(preds == targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    mena_loss = np.mean(losses)
    return correct_predictions, mena_loss


def bert_eval_epoch(model, valid_loader, loss_function, device='cuda'):
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for data in valid_loader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets = data["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)
            loss = loss_function(outputs.logits, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    val_loss = np.mean(losses)
    return correct_predictions, val_loss


def simple_train_epoch(model, optimizer, train_loader, loss_function, device='cuda'):
    losses = []
    correct_predictions = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        outputs = model(x)
        preds = torch.argmax(outputs, dim=1)
        loss = loss_function(outputs, y)

        correct_predictions += torch.sum(preds == y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    mena_loss = np.mean(losses)
    return correct_predictions, mena_loss


def simple_eval_epoch(model, valid_loader, loss_function, device='cuda'):
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            loss = loss_function(outputs, y)

            correct_predictions += torch.sum(preds == y)
            losses.append(loss.item())

    val_loss = np.mean(losses)
    return correct_predictions, val_loss
