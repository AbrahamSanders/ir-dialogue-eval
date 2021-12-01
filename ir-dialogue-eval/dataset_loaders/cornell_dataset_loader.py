"""
DatasetLoader implementation for the Cornell Movie-Dialogs dataset
"""
from os import path

from dataset_loaders.dataset_loader import DatasetLoader
from domain import Domain

class CornellMovieDialogsDatasetLoader(DatasetLoader):
    """DatasetLoader implementation for Cornell Movie-Dialogs dataset
    """
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        
    def _load_dataset(self):
        """ See docstring in base class.
        """
        # Import the dialogs from the dataset
        cornell_lines_filepath = path.join(self.dataset_path, "movie_lines.txt")
        cornell_conversations_filepath = path.join(self.dataset_path, "movie_conversations.txt")
        
        lines = {}
        with open(cornell_lines_filepath, encoding="utf-8", errors="ignore") as f:
            for line in f.readlines():
                line_fields = line.split(" +++$+++ ")
                if len(line_fields) == 5:
                    lines[line_fields[0]] = (line_fields[3], self._normalize_whitespace(line_fields[4]))
                    
        ids = []
        dialogs = []
        domains = []
        with open(cornell_conversations_filepath, encoding="utf-8") as f:
            for i, line in enumerate(f.readlines()):
                utterance_ids = line.split(" +++$+++ ")[-1].strip()[1:-1].replace("'", "").replace(" ", "")
                utterance_ids = utterance_ids.split(",")
                
                ids.append(i+1)
                dialogs.append([lines[uid] for uid in utterance_ids])
                domains.append([Domain.GENERAL.value])
                
        return ids, dialogs, domains