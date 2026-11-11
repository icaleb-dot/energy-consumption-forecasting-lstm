import copy

class EarlyStopping:
    def __init__(self, patience: int=3, min_delta: float=1e-5) -> None:
        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

        self.best_model_state = None



    def __call__(self, model, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Deepcopy para guardar el estado del mejor modelo
            # Aquí ya deja de ser None
            self.best_model_state = copy.deepcopy(model.state_dict())
        
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    
    def load_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
