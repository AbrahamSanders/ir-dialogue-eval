"""
DatasetLoader implementation for the MultiWOZ 2.2 dataset
"""

from dataset_loaders.sgd_dataset_loader import SGDDatasetLoader
from domain import Domain

class MultiWOZDatasetLoader(SGDDatasetLoader):
    """DatasetLoader implementation for MultiWOZ 2.2 dataset
    """
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        
        self.service_map_dict = {
            "hotel": Domain.HOTELS,
            "train": Domain.TRAIN,
            "attraction": Domain.ATTRACTION,
            "restaurant": Domain.RESTAURANTS,
            "hospital": Domain.HOSPITAL,
            "taxi": Domain.TAXI,
            "bus": Domain.BUSES,
            "police": Domain.POLICE
        }
        
    def _map_sgd_service(self, service):
        return self.service_map_dict[service].value