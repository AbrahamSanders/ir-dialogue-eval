"""
DatasetLoader implementation for the Frames dataset
"""
from os import path
import json
import re

from dataset_loaders.dataset_loader import DatasetLoader
from domain import Domain

class FramesDatasetLoader(DatasetLoader):
    """DatasetLoader implementation for Frames dataset
    """
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
    
    def _load_dataset(self):
        """ See docstring in base class.
        """
        # Import the dialogs from the dataset
        frames_filepath = path.join(self.dataset_path, "frames.json")
        with open(frames_filepath, encoding="utf-8") as f:
            frames_json = json.load(f)

        ids = [dialog["id"] for dialog in frames_json]
        
        dialogs = [[(utterance["author"], self._normalize_whitespace(utterance["text"])) for utterance in dialog["turns"]] 
                   for dialog in frames_json]
        
        domains = []
        for dialog in dialogs:
            dialog_text = " ".join([utt[1] for utt in dialog])
            domains.append([Domain.TRAVEL.value])
            if re.search("(?:fly|flight|airline|plane)", dialog_text, flags=re.IGNORECASE):
                domains[-1].append(Domain.FLIGHTS.value)
            if re.search("(?:hotel|motel|inn|resort|room|guest)", dialog_text, flags=re.IGNORECASE):
                domains[-1].append(Domain.HOTELS.value)

        return ids, dialogs, domains