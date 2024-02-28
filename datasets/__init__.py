from .phototourism import PhototourismDataset
from .phototourism_optimize import PhototourismOptimizeDataset
from .custom import CustomDataset
from .custom_optimize import CustomOptimizeDataset

dataset_dict = {
    "phototourism": PhototourismDataset,
    "phototourism_optimize": PhototourismOptimizeDataset,
    "custom": CustomDataset,
    "custom_optimize": CustomOptimizeDataset
}
