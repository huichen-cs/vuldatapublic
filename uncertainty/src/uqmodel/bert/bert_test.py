import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import pandas as pd
from typing import List, Union, Sequence

# change the default hugggingface cache directory
# os.environ['HF_DATASETS_CACHE'] = '~/.hfcache'
# os.environ['TRANSFORMERS_CACHE'] = '~/.hfcache'
# os.environ['HUGGINGFACE_HUB_CACHE'] = '~/.hfcache'
# os.environ['HF_HOME'] = '~/.hfcache'
cache_loc = os.path.expanduser("~/.hfcache")

# to run with hickory or tesla
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, num_classes):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        message = self.data.iloc[index]["commit_message"]
        patch = self.data.iloc[index]["commit_patch"]
        label = self.data.iloc[index]["label"]

        encoding = self.tokenizer.encode_plus(
            message,
            patch,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]
        labels = torch.tensor(label)
        # trunk-ignore(bandit/B101)
        assert len(input_ids) == len(attention_mask) == self.max_len
        return input_ids, attention_mask, labels


"""
A simple multiple layer classifier for estimating epistermic uncertainty via ensemble in pytorch
"""


class MultiLayerClassifierHead(torch.nn.Module):
    def __init__(
        self,
        input: int,
        output: int = 2,
        neurons: Sequence = [300, 300, 300],
        dropouts: Sequence = [0.25, 0.25, 0.25],
        activation=None,
    ):
        """Initialize a MultiLayerClassifierHead.

        Initialize a multi-layer neural network, each layer is a linear layer preceded with a dropout layer and
        activated by a RELU activation (by default). The class is organized like a Scikit-Learn classifier.
        """
        super().__init__()
        if len(neurons) != len(dropouts):
            raise ValueError("expected len(neurons) == len(dropouts")

        layers: List[
            Union[
                torch.nn.Dropout,
                torch.nn.Linear,
                torch.nn.ReLU,
                torch.nn.LeakyReLU,
                torch.nn.Module,
            ]
        ] = []
        for i in range(0, len(neurons)):
            if dropouts[i] is not None:
                layers.append(torch.nn.Dropout(dropouts[i]))
            if i == 0:
                layers.append(torch.nn.Linear(input, neurons[i]))
            elif i == len(neurons) - 1:
                layers.append(torch.nn.Linear(neurons[i - 1], output))
            else:
                layers.append(torch.nn.Linear(neurons[i - 1], neurons[i]))
            if activation is None:
                layers.append(torch.nn.ReLU())
            elif activation == "relu":
                layers.append(torch.nn.ReLU())
            elif activation == "leakyrelu":
                layers.append(torch.nn.LeakyReLU())
            elif isinstance(activation, torch.nn.Module):
                layers.append(activation)
            else:
                raise ValueError("activation is not a torch.nn.Module")

        self.layers = torch.nn.ModuleList(layers)
        self.n_layers = len(self.layers)

    def forward(self, x):
        shape = x.size()
        x = x.view(shape[0], -1)

        for i in range(self.n_layers):
            # print(i, self.layers[i])
            x = self.layers[i](x)
        return x


# class BertBinaryClassifier(nn.Module):
#     def __init__(self, dropout_prob=0.1):
#         super(BertBinaryClassifier, self).__init__()
#         self.bert = AutoModel.from_pretrained("microsoft/codebert-base", cache_dir=cache_loc)
#         for param in self.bert.parameters():
#             param.requires_grad = False
#         self.dropout = nn.Dropout(dropout_prob)
#         self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

#     def forward(self, input_ids, attention_mask):
#         # Feed input to BERT model
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs.pooler_output
#         # Apply dropout
#         pooled_output = self.dropout(pooled_output)
#         # Apply classifier
#         logits = self.classifier(pooled_output)
#         return logits


class BertBinaryClassifier(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super(BertBinaryClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(
            "microsoft/codebert-base", cache_dir=cache_loc
        )
        for param in self.bert.parameters():
            param.requires_grad = False
        print("self.bert.config.hidden_size = {}".format(self.bert.config.hidden_size))
        self.classifier = MultiLayerClassifierHead(
            self.bert.config.hidden_size,
            output=2,
            neurons=[1024, 1024, 1024],
            dropouts=[0.25, 0.25, 0.25],
            activation="leakyrelu",
        )
        # self.dropout = nn.Dropout(dropout_prob)
        # self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        # Feed input to BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        # # Apply dropout
        # pooled_output = self.dropout(pooled_output)
        # Apply classifier
        logits = self.classifier(pooled_output)
        return logits


def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    total_correct = 0
    for input_ids, attention_mask, targets in train_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, targets)
        total_loss += loss.item()

        _, preds = torch.max(logits, dim=1)
        total_correct += (preds == targets).sum().item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_correct / len(train_loader.dataset)
    return avg_loss, avg_acc


def create_dataset(pos, neg, neg_ratio=2):
    pos["label"] = 1
    neg["label"] = 0
    pn_df = pd.concat([pos, neg], ignore_index=True)
    pn_df["cve_commit"] = pn_df["cve"] + "_" + pn_df["commit"]
    print(pn_df)

    def func(x):
        return pd.concat([x[x["label"] == 1], x[x["label"] == 0].head(neg_ratio)])

    # func = lambda x: pd.concat([x[x['label'] == 1], x[x['label'] == 0].head(neg_ratio)])
    df_selected = pn_df.groupby("cve_commit").apply(func)
    df_selected = df_selected.reset_index(drop=True)
    print(df_selected)
    return df_selected


# def test(model, test_loader):
#     model.eval()
#
#     # Initialize variables for computing metrics
#     total_correct = 0
#     total_samples = 0
#
#     # Loop over the test data
#     for batch in test_loader:
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['label'].to(device)
#
#         # Compute the output logits
#         with torch.no_grad():
#             logits = model(input_ids, attention_mask)
#
#         # Convert logits to probabilities
#         probs = torch.softmax(logits, dim=1)
#
#         # Compute predicted class for each example
#         preds = torch.argmax(probs, dim=1)
#
#         # Update metrics
#         total_correct += (preds == labels).sum().item()
#         total_samples += len(labels)
#
#     # Compute evaluation metrics
#     accuracy = total_correct / total_samples
#     # precision = ...
#     # recall = ...
#     # f1_score = ...
#
#     print(f"Accuracy: {accuracy:.4f}")
#     # print(f"Precision: {precision:.4f}")
#     # print(f"Recall: {recall:.4f}")
#     # print(f"F1 score: {f1_score:.4f}")


def main():
    data_dirpath = "methods/VCMatch/data/SAP/"
    pos = pd.read_csv(
        os.path.join(data_dirpath, "SAP_full_commits.csv"),
        index_col=0,
        keep_default_na=False,
    )
    neg = pd.read_csv(
        os.path.join(data_dirpath, "SAP_negative_commits_10x.csv"),
        index_col=0,
        keep_default_na=False,
    )
    data = create_dataset(pos, neg)

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/codebert-base", cache_dir=cache_loc
    )
    max_len = 512
    num_classes = 2

    dataset = TextClassificationDataset(data, tokenizer, max_len, num_classes)
    batch_size = 6
    num_epochs = 10

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BertBinaryClassifier()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, loss_fn, device)
        print(
            f"Epoch {epoch+1}: Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}"
        )

    # test(model)


if __name__ == "__main__":
    main()
