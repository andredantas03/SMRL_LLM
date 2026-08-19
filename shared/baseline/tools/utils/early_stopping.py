import torch
import os

class EarlyStopping:
    """
    Early stopping para minimizar uma métrica (ex: val_loss).
    Salva automaticamente o melhor checkpoint.
    """

    def __init__(
        self,
        patience=5,
        min_delta=0.0,
        checkpoint_path="best_model.pt",
        verbose=True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose

        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, metric, model):
        """
        metric: valor da métrica de validação (float)
        model: torch.nn.Module
        """
        score = -metric  # porque queremos minimizar

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)
            return

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping: sem melhora "
                    f"({self.counter}/{self.patience})"
                )
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self._save_checkpoint(model)

    def _save_checkpoint(self, model):
        torch.save(model.state_dict(), self.checkpoint_path)
        if self.verbose:
            print("EarlyStopping: melhor modelo salvo.")