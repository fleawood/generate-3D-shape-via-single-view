from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import json


class DepthDataset(Dataset):
    def __init__(self, index_filename):
        with open(index_filename, "r") as file:
            self.index = json.load(file)
        # self.num_cats = num_cats

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        files = self.index[item][0]
        images = [Image.open(os.path.join(file)).convert("L") for file in files]
        all_data = np.stack(images, axis=0).astype(np.float32) / 255.0
        num_views = 20
        view_id = np.random.randint(num_views)
        single_data = all_data[view_id: view_id + 1, :, :]
        label = self.index[item][1]
        return {'all_data': all_data, 'single_data': single_data, 'label': label}
