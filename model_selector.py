
class ModelSelector:

    def __init__(self, threshold = 0.1):
        self.models = {} # Stores {"epochX": {"model": ..., "loss": ...}}
        self.threshold = threshold
        self.lowest_loss = float("inf")

    def add(self, epoch, model_state, loss):
        key = f"epoch{epoch}"

        self.models[key] = {
            "model": model_state,
            "loss": loss
        }

        if loss < self.lowest_loss:
            self.lowest_loss = loss
        
        self._remove()

    def _remove(self):
        keys_to_delete = []

        for key in self.models:
            if self.models[key]["loss"] > self.lowest_loss * (1 + self.threshold):
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self.models[key]
        

    def return_best_model(self):
        for epoch_number in self.models:
            return epoch_number, self.models[epoch_number]["model"], self.models[epoch_number]["loss"]
        

    