"""
DatasetLoader implementation for the DailyDialog dataset
"""
from os import path

from dataset_loaders.dataset_loader import DatasetLoader
from domain import Domain

class DailyDialogDatasetLoader(DatasetLoader):
    """DatasetLoader implementation for DailyDialog dataset
    """
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        
    def _load_dataset(self):
        """ See docstring in base class.
        """
        # Import the dialogs from the dataset
        dailydialog_filepath = path.join(self.dataset_path, "dialogues_text.txt")
        
        ids = []
        dialogs = []
        domains = []
        with open(dailydialog_filepath, encoding="utf-8") as f:
            for i, dialog in enumerate(f.readlines()):
                ids.append(i+1)
                dialogs.append([("Speaker %s" % ((j % 2)+1), self._clean_tokenization(utt)) 
                                for j, utt in enumerate(dialog.split("__eou__")[:-1])])
                domains.append([Domain.GENERAL.value])
        
        return ids, dialogs, domains
                
    def _clean_tokenization(self, text):
        text = text.replace("’", "'")
        text = text.replace(" ' ", "'")
        text = text.replace(" ? ", "? ")
        text = text.replace(" ... ",  "... ")
        text = text.replace(" .. . ",  "... ")
        text = text.replace(" .. ",  ".. ")
        text = text.replace(" . ",  ". ")
        text = text.replace(" ! ", "! ")
        text = text.replace(" , ", ", ")
        text = text.replace(" $ ", " $")
        text = text.replace(" % ", "% ")
        text = text.replace(" # ", " #")
        text = text.replace(" ( ", " (")
        text = text.replace(" ) ", ") ")
        text = text.replace(" / ", "/")
        text = text.replace("\\", "")
        text = self._normalize_whitespace(text)
        return text