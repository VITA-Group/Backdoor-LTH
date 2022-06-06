
import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

assert False
# Not ready

class PoisonedGTSRB(Dataset):
    base_folder = 'GTSRB'

    def __init__(self, root, transform=None, train=True, poison_ratio=0.01, target=0, patch_size=5, random_loc=False, black_trigger=False):
        """
        Args:
            train (bool): Load trainingset or test set.
            root_dir (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root

        normalize = transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        self.transform = transforms.Compose([
                transforms.Resize([32,32]),
                transforms.ToTensor(),
                normalize
            ])

        self.sub_directory = 'trainingset' if train else 'testset'
        self.csv_file_name = 'training.csv' if train else 'test.csv'

        csv_file_path = os.path.join(
            root, self.base_folder, self.sub_directory, self.csv_file_name)

        self.csv_data = pd.read_csv(csv_file_path)

        self.imgs = torch.empty([len(self.csv_data), 3, 32, 32])
        self.labels = torch.empty([len(self.csv_data)], dtype=torch.long)
        for idx in range(len(self.csv_data)):
            
            img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory,
                                    self.csv_data.iloc[idx, 0])
            self.imgs[idx] = self.transform(Image.open(img_path))

            self.labels[idx] = torch.tensor(self.csv_data.iloc[idx, 1])

        idx = torch.randperm(len(self.csv_data))
        self.imgs = self.imgs[idx,...]
        self.labels = self.labels[idx]

        trans_trigger = transforms.Compose(
            [transforms.Resize((patch_size, patch_size)), transforms.ToTensor()]
        )

        if not black_trigger:
            trigger = Image.open("dataset/triggers/htbd.png").convert("RGB")
        else:
            trigger = Image.open("triggers/clbd.png").convert("RGB")

        trigger = trans_trigger(trigger)
        for i in range(0, int(len(self.imgs) * poison_ratio)):
            if not random_loc:
                start_x = 32 - patch_size - 3
                start_y = 32 - patch_size - 3
            else:
                start_x = random.randint(0, 32 - patch_size)
                start_y = random.randint(0, 32 - patch_size)
            self.imgs[i, :, start_x: start_x + patch_size, start_y: start_y + patch_size] = trigger
            self.labels[i] = target

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):

        return self.imgs[idx], self.labels[idx]
