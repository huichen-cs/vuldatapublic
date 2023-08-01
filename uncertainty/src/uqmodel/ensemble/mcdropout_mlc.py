"""
A simple multiple layer classifier with dropouts in pytorch.
"""

import torch


def train(
    model, train_dataset, val_dataset, model_chkpt_path, num_epochs=20, device=None
):
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-03)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # train phase
        train_loss = []
        for train_batch in train_dataset:
            x, y = train_batch
            x = x.to(device)
            y = y.to(device)
            logits = model(x)

            loss = loss_func(logits, y)
            train_loss.append(loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss = torch.mean(torch.tensor(train_loss))

        # validation phase
        with torch.no_grad():
            val_loss = []
            for val_batch in val_dataset:
                x, y = val_batch
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                val_loss.append(loss_func(logits, y))
            val_loss = torch.mean(torch.tensor(val_loss))
            print(
                "epoch:",
                epoch,
                "train_loss: ",
                train_loss.item(),
                "val_loss: ",
                val_loss.item(),
            )
    torch.save(model, model_chkpt_path)


def predict_class(model, x):
    model = model.eval()
    outputs = model(x)
    _, pred = torch.max(outputs.data, 1)
    model = model.train()
    return pred


def predict(model, x, n_samples=1000):
    predicted_class = predict_class(model, x)
    # trunk-ignore(bandit/B101)
    assert model.training  # this is the key -- turn on dropouts when doing inference
    logits = []
    proba = []
    for _ in range(n_samples):
        y_logits = model(x)
        y_proba = torch.nn.functional.softmax(logits, dim=1)
        logits.append(y_logits)
        proba.append(y_proba)
    return predicted_class, logits, proba
