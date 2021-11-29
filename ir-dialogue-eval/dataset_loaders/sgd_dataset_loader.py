"""
DatasetLoader implementation for the SGD dataset
"""
from os import path
from glob import glob
import json

from dataset_loaders.dataset_loader import DatasetLoader

class SGDDatasetLoader(DatasetLoader):
    """DatasetLoader implementation for Schema Guided Dialogue dataset
    """
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        
    def _load_dataset(self):
        """ See docstring in base class.
        """
        # Import the dialogs from the dataset
        ids = []
        dialogs = []
        domains = []
        for sub_dir in ["train", "dev", "test"]:
            glob_path = path.join(self.dataset_path, sub_dir, "dialogues_*.json")
            sgd_filepaths = glob(glob_path)
            
            for sgd_filepath in sgd_filepaths:
                with open(sgd_filepath) as f:
                    sgd_json = json.load(f)
                
                ids.extend(["%s_%s" % (sub_dir, dialog["dialogue_id"]) for dialog in sgd_json])
                
                dialogs.extend([[(utterance["speaker"], self._normalize_whitespace(utterance["utterance"])) 
                                 for utterance in dialog["turns"]] for dialog in sgd_json])
                
                domains.extend([list(map(self._map_sgd_service, dialog["services"])) for dialog in sgd_json])
                
        return ids, dialogs, domains
                
    def _map_sgd_service(self, service):
        return service.split("_")[0]