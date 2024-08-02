from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


class EarlyStopping(object):
    """Early stoper for model."""

    def __init__(self, patience=5, min_delta=0, min_loss=None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.min_loss = min_loss

    def reset(self) -> EarlyStopping:
        self.min_loss = self.init_min_loss
        self.counter = 0
        self.early_stop = False
        return self

    def __call__(self, loss):
        logger.debug(
            f"EarlyStopping: min loss: {self.min_loss} loss: {loss} paitence: {self.patience} counter: {self.counter}"
        )
        if self.min_loss and ((loss - self.min_loss) >= self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
        if not self.min_loss or loss < self.min_loss:
            self.min_loss = loss
        return self.early_stop
