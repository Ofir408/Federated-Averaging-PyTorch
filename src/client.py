import gc
import pickle
import logging

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """

    def __init__(self, client_id, local_data, device):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.data = local_data
        self.device = device
        self.__model = None

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.dataloader = DataLoader(self.data, batch_size=client_config["batch_size"], shuffle=True)
        self.local_epoch = client_config["num_local_epochs"]
        self.criterion = client_config["criterion"]
        self.optimizer = client_config["optimizer"]
        self.optim_config = client_config["optim_config"]

    def client_update(self):
        """Update local model using local dataset."""
        self.model.train()
        self.model.to(self.device)

        optimizer = eval(self.optimizer)(self.model.parameters(), **self.optim_config)
        for e in range(self.local_epoch):
            for step, batch in enumerate(self.dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                age_ids, input_ids, posi_ids, segment_ids, attMask, masked_label = batch
                loss, pred, label = self.model(input_ids, age_ids, segment_ids, posi_ids, attention_mask=attMask,
                                               masked_lm_labels=masked_label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

    def client_evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for step, batch in enumerate(self.dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                age_ids, input_ids, posi_ids, segment_ids, attMask, masked_label = batch
                loss, pred, label = self.model(input_ids, age_ids, segment_ids, posi_ids, attention_mask=attMask,
                                               masked_lm_labels=masked_label)
                test_loss += loss
                predicted = pred.argmax(dim=1, keepdim=True) # TODO SHOULD BE CHANGED FOR MULTI-LABEL TASKS!
                correct += predicted.eq(label.view_as(predicted)).sum().item()

                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\n"
        print(message, flush=True)
        logging.info(message)
        del message
        gc.collect()

        return test_loss, test_accuracy
