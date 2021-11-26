"""
DatasetLoader class.
"""

import abc
from os import path

class DatasetLoader(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset_name = path.basename(dataset_path)
    
    @abc.abstractmethod
    def _load_dataset(self):
        """Subclass must implement this.
        """
        pass
    
    @abc.abstractmethod
    def _filter_dialog(self, id_, dialog):
        """Subclass can optionally implement this.
        """
        return False
    
    def load_dataset(self, min_dialog_utterances=2):
        # Load the dialogs
        ids, dialogs, domains = self._load_dataset()
        
        # Filter the dialogs
        skipped_indices = []
        filtered_dialog_min_utterances = 0
        filtered_dataset_specific = 0
        for i, dialog in enumerate(dialogs):
            # min utterances filter
            if min_dialog_utterances and len(dialog) < min_dialog_utterances:
                filtered_dialog_min_utterances += 1
                skipped_indices.append(i)
                continue
            
            # subclass-provided filter
            if self._filter_dialog(ids[i], dialog):
                filtered_dataset_specific += 1
                skipped_indices.append(i)
                continue
        
        # Remove skipped dialogs and values
        for skipped_idx in reversed(skipped_indices):
            ids.pop(skipped_idx)
            dialogs.pop(skipped_idx)
            domains.pop(skipped_idx)
            
        if min_dialog_utterances:
            print ("Skipped %d dialogs which don't have at least %d utterances." 
                   % (filtered_dialog_min_utterances, min_dialog_utterances))

        print ("Skipped %d dialogs which do not pass the dataset-specific filter." 
               % filtered_dataset_specific)
            
        return ids, dialogs, domains
        