import torch
import random
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import CIFAR100


class PoisonedCIFAR100(data.Dataset):
    def __init__(self, root,
                train=True, 
                poison_ratio=0.1, 
                target=0, 
                patch_size=5,
                random_loc=False, 
                upper_right=True,
                bottom_left=False,
                augmentation=True, 
                black_trigger=False):

        self.train = train
        self.poison_ratio = poison_ratio
        self.root = root

        if random_loc:
            print('Using random location')
        if upper_right:
            print('Using fixed location of Upper Right')
        if bottom_left:
            print('Using fixed location of Bottom Left')

        # init trigger
        trans_trigger = transforms.Compose(
            [transforms.Resize((patch_size, patch_size)), transforms.ToTensor(), lambda x: x * 255]
        )
        trigger = Image.open("dataset/triggers/htbd.png").convert("RGB")
        if black_trigger:
            print('Using black trigger')
            trigger = Image.open("dataset/triggers/clbd.png").convert("RGB")
        trigger = trans_trigger(trigger)
        trigger = torch.tensor(np.transpose(trigger.numpy(), (1, 2, 0))) # 5,5,3 [0,255]

        normalize = transforms.Normalize(mean = (0.5071, 0.4866, 0.4409), std = (0.2673, 0.2564, 0.2762))

        if augmentation and self.train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    normalize
                ])

        if self.train:
            dataset = CIFAR100(root, train=True, transform=self.transform, download=True)
        else:
            dataset = CIFAR100(root, train=False, transform=self.transform, download=True)

        self.imgs = dataset.data
        self.labels = dataset.targets

        image_size = self.imgs.shape[1]
        for i in range(0, int(len(self.imgs) * poison_ratio)):
            
            if random_loc:
                start_x = random.randint(0, image_size - patch_size)
                start_y = random.randint(0, image_size - patch_size)
            elif upper_right:
                start_x = image_size - patch_size - 3
                start_y = image_size - patch_size - 3
            elif bottom_left:
                start_x = 3
                start_y = 3
            else:
                assert False

            self.imgs[i][start_x: start_x + patch_size, start_y: start_y + patch_size, :] = trigger
            self.labels[i] = target
        self.imgs = torch.tensor(np.transpose(self.imgs, (0,3,1,2))) # Batch-size, 3, 32, 32

    def __getitem__(self, index):
        img = self.transform(self.imgs[index])
        return img, self.labels[index]

    def __len__(self):
        return len(self.imgs)


