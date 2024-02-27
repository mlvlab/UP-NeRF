from .phototourism import PhototourismDataset
from .phototourism_optimize import PhototourismOptimizeDataset
from .custom import CustomDataset

dataset_dict = {
    "phototourism": PhototourismDataset,
    "phototourism_optimize": PhototourismOptimizeDataset,
    "custom": CustomDataset
}
