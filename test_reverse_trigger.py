from __future__ import print_function

import argparse
import os
import random

import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.utils.data import DataLoader, Subset

from dataset.clean_label_cifar10 import CleanLabelPoisonedCIFAR10
from dataset.poisoned_cifar10 import PoisonedCIFAR10
from dataset.poisoned_cifar100 import PoisonedCIFAR100
from dataset.poisoned_rimagenet import RestrictedImageNet

from models.adv_resnet import resnet20s as robust_res20s
from models.densenet import *
from models.model_zoo import *
from models.resnets import resnet20s
from models.vgg import *
from pruner import *
from utils_reverse_trigger import *
from new_test_reverse_trigger import remask_tabor

# Settings
parser = argparse.ArgumentParser(description='PyTorch Analysis')

##################################### Backdoor #################################################
parser.add_argument("--patch_size", type=int, default=5, help="Size of the patch")
parser.add_argument("--freq", dest="freq", action="store_true", help="Hidden trigger mode or normal mode?")
parser.add_argument("--random_loc", dest="random_loc", action="store_true",
                    help="Is the location of the trigger randomly selected or not?")
parser.add_argument("--target", default=0, type=int, help="The target class")
parser.add_argument("--upper_right", dest="upper_right", action="store_true")
parser.add_argument("--bottom_left", dest="bottom_left", action="store_true")
parser.add_argument("--black_trigger", action="store_true")
parser.add_argument("--clean_label_attack", action="store_true")
parser.add_argument('--robust_model', type=str, default=None, help='checkpoint file')
parser.add_argument("--poison_ratio", type=float, default=0.0)
parser.add_argument("--noise_image", dest="noise_image", action="store_true")
parser.add_argument("--image_number", type=float, default=10)

parser.add_argument("--fixmask", action="store_true")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--data_number', type=int, default=10, help='number of images')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--batch_size', type=int, default=200, help='batch size')
parser.add_argument('--workers', type=int, default=2, help='number of workers in dataloader')

parser.add_argument('--arch', type=str, default='resnet20', help='architecture')
parser.add_argument('--pretrained', type=str, default=None, help='pretrained weight')
parser.add_argument('--save_dir', type=str, default=None, help='mark-mask direction')
parser.add_argument('--name', type=str, default=None, help='mark-mask file name')

parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID')

args = parser.parse_args()

torch.cuda.set_device(args.gpu)

# set random seed to reproduce the work
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True

for arg in vars(args):
    print(arg, getattr(args, arg))

os.makedirs(args.save_dir, exist_ok=True)

#############################################################################
####################  Get the hessian data and model  #######################
#############################################################################
img_shape = (3,32,32)

# prepare dataset
if args.dataset == 'cifar10':
    print('Dataset = CIFAR10')
    classes = 10
    if args.clean_label_attack:
        print('Clean Label Attack')
        robust_model = robust_res20s(num_classes=classes)
        robust_weight = torch.load(args.robust_model, map_location='cpu')
        if 'state_dict' in robust_weight.keys():
            robust_weight = robust_weight['state_dict']
        robust_model.load_state_dict(robust_weight)
        train_set = CleanLabelPoisonedCIFAR10(args.data, poison_ratio=args.poison_ratio, patch_size=args.patch_size,
                                            random_loc=args.random_loc, upper_right=args.upper_right,
                                            bottom_left=args.bottom_left,
                                            target=args.target, black_trigger=args.black_trigger,
                                            robust_model=robust_model)
    else:
        train_set = PoisonedCIFAR10(args.data, train=True, poison_ratio=args.poison_ratio, patch_size=args.patch_size,
                                    random_loc=args.random_loc, upper_right=args.upper_right,
                                    bottom_left=args.bottom_left,
                                    target=args.target, black_trigger=args.black_trigger)

    sub_train_set = Subset(train_set, list(range(50000))[:args.data_number])

    clean_testset = PoisonedCIFAR10(args.data, train=False, poison_ratio=0, patch_size=args.patch_size,
                                    random_loc=args.random_loc, upper_right=args.upper_right,
                                    bottom_left=args.bottom_left,
                                    target=args.target, black_trigger=args.black_trigger)
    poison_testset = PoisonedCIFAR10(args.data, train=False, poison_ratio=1, patch_size=args.patch_size,
                                    random_loc=args.random_loc, upper_right=args.upper_right,
                                    bottom_left=args.bottom_left,
                                    target=args.target, black_trigger=args.black_trigger)
    train_dl = DataLoader(sub_train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                          pin_memory=True)
    clean_test_dl = DataLoader(clean_testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                               pin_memory=True)
    poison_test_dl = DataLoader(poison_testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                pin_memory=True)
elif args.dataset == 'cifar100':
    print('Dataset = CIFAR100')
    classes = 100
    train_set = PoisonedCIFAR100(args.data, train=True, poison_ratio=args.poison_ratio, patch_size=args.patch_size,
                                 random_loc=args.random_loc, upper_right=args.upper_right, bottom_left=args.bottom_left,
                                 target=args.target, black_trigger=args.black_trigger)
    clean_testset = PoisonedCIFAR100(args.data, train=False, poison_ratio=0, patch_size=args.patch_size,
                                     random_loc=args.random_loc, upper_right=args.upper_right,
                                     bottom_left=args.bottom_left,
                                     target=args.target, black_trigger=args.black_trigger)
    poison_testset = PoisonedCIFAR100(args.data, train=False, poison_ratio=1, patch_size=args.patch_size,
                                      random_loc=args.random_loc, upper_right=args.upper_right,
                                      bottom_left=args.bottom_left,
                                      target=args.target, black_trigger=args.black_trigger)
    sub_train_set = Subset(train_set, list(range(50000))[:args.data_number])
    train_dl = DataLoader(sub_train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                          pin_memory=True)
    clean_test_dl = DataLoader(clean_testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                               pin_memory=True)
    poison_test_dl = DataLoader(poison_testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                pin_memory=True)
elif args.dataset == 'rimagenet':
    img_shape = (3,224,224)
    print('Dataset = Restricted ImageNet')
    classes = 9
    dataset = RestrictedImageNet(args.data)
    train_dl, _, _ = dataset.make_loaders(workers=args.workers, batch_size=args.batch_size,
                                          poison_ratio=args.poison_ratio, target=args.target,
                                          patch_size=args.patch_size, black_trigger=args.black_trigger, subset=args.data_number)
    _, clean_test_dl = dataset.make_loaders(only_val=True, workers=args.workers, batch_size=args.batch_size,
                                            poison_ratio=0, target=args.target, patch_size=args.patch_size,
                                            black_trigger=args.black_trigger)
    _, poison_test_dl = dataset.make_loaders(only_val=True, workers=args.workers, batch_size=args.batch_size,
                                             poison_ratio=1, target=args.target, patch_size=args.patch_size,
                                             black_trigger=args.black_trigger)
else:
    raise ValueError('Unknow Datasets')

# prepare model
if args.dataset == 'rimagenet':
    if args.arch == 'resnet18':
        model = models.resnet18(num_classes=classes)
    else:
        raise ValueError('Unknow architecture')
else:
    if args.arch == 'resnet18':
        model = ResNet18(num_classes=classes)
    elif args.arch == 'resnet20':
        model = resnet20s(num_classes=classes)
    elif args.arch == 'densenet100':
        model = densenet_100_12(num_classes=classes)
    elif args.arch == 'vgg16':
        model = vgg16_bn(num_classes=classes)
    else:
        raise ValueError('Unknow architecture')

#############################################################################
##############################  Get the model  ##############################
#############################################################################


model.cuda()

if args.pretrained:
    print('===> loading weight from {} <==='.format(args.pretrained))
    pretrained_weight = torch.load(args.pretrained, map_location='cuda')

    if 'state_dict' in pretrained_weight:
        pretrained_weight = pretrained_weight['state_dict']
    sparse_mask = extract_mask(pretrained_weight)
    if len(sparse_mask) > 0:
        prune_model_custom(model, sparse_mask)
    model.load_state_dict(pretrained_weight)
    check_sparsity(model)

model.eval()

save_mark = {}
save_mask = {}



for target_label in range(classes):

    mark_best, mask_best = remask_tabor(dir=args.save_dir, model=model,
                                        dataloader=train_dl, test_dataloader=clean_test_dl,
                                        target_label=target_label,
                                        random_noise=args.noise_image, img_shape=img_shape)

    save_mark[target_label] = mark_best
    save_mask[target_label] = mask_best

    norm = mask_best.norm(p=1)
    ASR = validate(clean_test_dl, model, mark_best, mask_best, target_label, hard_mask_trigger=True, patch_size=args.patch_size)
    # validate(train_dl, model, mark_best, mask_best, target_label)
    print('Target = {}, Norm = {:.1f}, ASR = {:1f}'.format(target_label+1, norm, ASR*100))


all_mask_mark = {}
all_mask_mark['Mark'] = save_mark
all_mask_mark['Mask'] = save_mask

all_path = os.path.join(args.save_dir, "{}_all_mask_mark.pth".format(args.name))
torch.save(all_mask_mark, all_path)
