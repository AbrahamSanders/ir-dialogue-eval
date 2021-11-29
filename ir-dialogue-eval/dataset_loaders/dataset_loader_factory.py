"""
DatasetLoader factory
"""
from os import path
from dataset_loaders.frames_dataset_loader import FramesDatasetLoader
from dataset_loaders.sgd_dataset_loader import SGDDatasetLoader

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
    dataset_name = path.basename(dataset_path)
    
    if dataset_name == "frames":
        return FramesDatasetLoader(dataset_path)
    if dataset_name == "dstc8-schema-guided-dialogue":
        return SGDDatasetLoader(dataset_path)
        
    raise ValueError("There is no DatasetLoader implementation for '%s'. Please add one!" % dataset_name)