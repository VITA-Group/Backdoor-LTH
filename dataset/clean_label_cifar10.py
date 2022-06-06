from PIL import Image
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import CIFAR10
import numpy as np
import torch
import random
from dataset.pgd_attack import PgdAttack


class CleanLabelPoisonedCIFAR10(data.Dataset):

    def __init__(self, root, 
                    transform=None,
                    poison_ratio=0.1, 
                    target=0, 
                    patch_size=5, 
                    random_loc=False, 
                    upper_right=True,
                    bottom_left=False,
                    augmentation=True, 
                    black_trigger=False,
                    pgd_alpha: float = 2 / 255,
                    pgd_eps: float = 8 / 255, 
                    pgd_iter=7, 
                    robust_model=None):

        self.root = root
        self.poison_ratio = poison_ratio
        self.target_label = target
        self.patch_size = patch_size
        self.random_loc = random_loc
        self.upper_right = upper_right
        self.bottom_left = bottom_left
        self.pgd_alpha = pgd_alpha
        self.pgd_eps = pgd_eps
        self.pgd_iter = pgd_iter
        self.model = robust_model
        self.attacker = PgdAttack(self.model, self.pgd_eps, self.pgd_iter, self.pgd_alpha)

        if random_loc:
            print('Using random location')
        if upper_right:
            print('Using fixed location of Upper Right')
        if bottom_left:
            print('Using fixed location of Bottom Left')

        # init trigger
        trans_trigger = transforms.Compose(
            [transforms.Resize((patch_size, patch_size)), transforms.ToTensor()]
        )
        trigger = Image.open("dataset/triggers/htbd.png").convert("RGB")
        if black_trigger:
            print('Using black trigger')
            trigger = Image.open("dataset/triggers/clbd.png").convert("RGB")
        self.trigger = trans_trigger(trigger)

        normalize = transforms.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2470, 0.2435, 0.2616))

        if pgd_alpha is None:
            pgd_alpha = 1.5 * pgd_eps / pgd_iter
        self.pgd_alpha: float = pgd_alpha
        self.pgd_eps: float = pgd_eps
        self.pgd_iter: int = pgd_iter

        if augmentation:
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

        dataset = CIFAR10(root, train=True, download=True)

        self.imgs = dataset.data
        self.labels = dataset.targets
        self.image_size = self.imgs.shape[1]

        if self.poison_ratio != 0.0:
            self.imgs = torch.tensor(np.transpose(self.imgs, (0, 3, 1, 2)), dtype=torch.float32) / 255.
            target_index, other_index = self.separate_img()
            self.poison_num = int(len(target_index) * self.poison_ratio)
            target_imgs = self.imgs[target_index[:self.poison_num]]
            target_imgs = self.attacker(target_imgs, self.target_label * torch.ones(len(target_imgs), dtype=torch.long)) # (N,3,32,32)         
            target_imgs = self.add_trigger(target_imgs)
            self.imgs[target_index[:self.poison_num]] = target_imgs
            print('poison images = {}'.format(self.poison_num))
        else:
            print("Point ratio is zero!")

    def __getitem__(self, index):
        img = self.transform(self.imgs[index])
        return img, self.labels[index]

    def __len__(self):
        return len(self.imgs)

    def separate_img(self):
        """
        Collect all the images, which belong to the target class
        """
        dataset = CIFAR10(self.root, train=True, download=True)
        target_img_index = []
        other_img_index = []
        all_data = dataset.data
        all_label = dataset.targets
        for i in range(len(all_data)):
            if self.target_label == all_label[i]:
                target_img_index.append(i)
            else:
                other_img_index.append(i)
        return torch.tensor(target_img_index), torch.tensor(other_img_index)

    def add_trigger(self, img):

        if self.random_loc:
            start_x = random.randint(0, self.image_size - self.patch_size)
            start_y = random.randint(0, self.image_size - self.patch_size)
        elif self.upper_right:
            start_x = self.image_size - self.patch_size - 3
            start_y = self.image_size - self.patch_size - 3
        elif self.bottom_left:
            start_x = 3
            start_y = 3
        else:
            assert False

        img[:, :, start_x: start_x + self.patch_size, start_y: start_y + self.patch_size] = self.trigger
        return img

