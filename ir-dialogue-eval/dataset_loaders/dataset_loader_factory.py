"""
DatasetLoader factory
"""
from os import path
from dataset_loaders.frames_dataset_loader import FramesDatasetLoader
from dataset_loaders.sgd_dataset_loader import SGDDatasetLoader
from dataset_loaders.multiwoz_dataset_loader import MultiWOZDatasetLoader
from dataset_loaders.dailydialog_dataset_loader import DailyDialogDatasetLoader

def get_dataset_loader(dataset_path):
    """
    Returns an instance of the appropriate DatasetLoader class for the given dataset.
    The dataset name is the base directory name provided.

    Parameters
    ----------
    dataset_path : str
        The directory of the dataset.

    Raises
    ------
    ValueError
        This error occurs if no DatasetLoader implementation exists for the specified dataset.
    """
    dataset_name = path.basename(dataset_path).lower()
    
    if dataset_name == "frames":
        return FramesDatasetLoader(dataset_path)
    if dataset_name == "dstc8-schema-guided-dialogue":
        return SGDDatasetLoader(dataset_path)
    if dataset_name == "multiwoz_2.2":
        return MultiWOZDatasetLoader(dataset_path)
    if dataset_name == "ijcnlp_dailydialog":
        return DailyDialogDatasetLoader(dataset_path)
    
    raise ValueError("There is no DatasetLoader implementation for '%s'. Please add one!" % dataset_name)