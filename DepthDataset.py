from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import json


class DepthDataset(Dataset):
    def __init__(self, index_filename, category):
        with open(index_filename, "r") as file:
            self.index = json.load(file)[category]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        files = self.index[item]
        images = [Image.open(os.path.join(file)).convert("L") for file in files]
        real_all_views = np.stack(images, axis=0).astype(np.float32) / 255.0
        num_views = 20
        view_id = np.random.randint(num_views)
        real_single_view = real_all_views[view_id: view_id + 1, :, :]
        return {'real_all_views': real_all_views, 'real_single_view': real_single_view}
