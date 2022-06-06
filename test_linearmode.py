
import os
import sys
import json
import time 
import random 
import argparse
import numpy as np

import copy 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from advertorch.utils import NormalizeByChannelMeanStd


from pruner import *
from dataset.poisoned_cifar10 import PoisonedCIFAR10
from dataset.poisoned_cifar100 import PoisonedCIFAR100
from dataset.poisoned_rimagenet import RestrictedImageNet
from dataset.clean_label_cifar10 import CleanLabelPoisonedCIFAR10

from models.resnets import resnet20s
# ResNet18
from models.model_zoo import *
from models.densenet import *
from models.vgg import *
from models.adv_resnet import resnet20s as robust_res20s

from utils_linear_mode import *

# Settings
parser = argparse.ArgumentParser(description='PyTorch pyhessian analysis')
##################################### Backdoor #################################################
parser.add_argument("--poison_ratio", type=float, default=0.01)
parser.add_argument("--patch_size", type=int, default=5, help="Size of the patch")
parser.add_argument("--random_loc", dest="random_loc", action="store_true", help="Is the location of the trigger randomly selected or not?")
parser.add_argument("--upper_right", dest="upper_right", action="store_true")
parser.add_argument("--bottom_left", dest="bottom_left", action="store_true")
parser.add_argument("--target", default=0, type=int, help="The target class")
parser.add_argument("--black_trigger", action="store_true")
parser.add_argument("--clean_label_attack", action="store_true")
parser.add_argument('--robust_model', type=str, default=None, help='checkpoint file')
parser.add_argument('--save_file', default=None, type=str)
parser.add_argument("--init", action="store_true")
parser.add_argument("--compare_retrain", action="store_true")
parser.add_argument('--max', type=int, default=49, help='checkpoint_number')

##################################### Dataset #################################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--input_size', type=int, default=32, help='size of input images')
parser.add_argument('--rate', default=0.2, type=float, help='pruning rate')
parser.add_argument('--lr', default=1e-3, type=float, help='pruning rate')
##################################### General setting ############################################
parser.add_argument('--arch', type=str, default='resnet18', help='network architecture')
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--workers', type=int, default=2, help='number of workers in dataloader')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--batch_num', type=int, default=None, help='batch number')
parser.add_argument('--pretrained_dir', type=str, default=None, help='pretrained weight')
parser.add_argument('--finetune_iter', type=int, default=0, help='batch number')

def main():
    global args
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))

    torch.cuda.set_device(int(args.gpu))
    if args.seed:
        setup_seed(args.seed)

    # prepare dataset
    if args.dataset == 'cifar10':
        print('Dataset = CIFAR10')
        classes = 10
        if args.clean_label_attack:
            print('Clean Label Attack')
            robust_model = robust_res20s(num_classes = classes)
            robust_weight = torch.load(args.robust_model, map_location='cpu')
            if 'state_dict' in robust_weight.keys():
                robust_weight = robust_weight['state_dict']
            robust_model.load_state_dict(robust_weight)
            train_set = CleanLabelPoisonedCIFAR10(args.data, poison_ratio=args.poison_ratio, patch_size=args.patch_size,
                                    random_loc=args.random_loc, upper_right=args.upper_right, bottom_left=args.bottom_left, 
                                    target=args.target, black_trigger=args.black_trigger, robust_model=robust_model)
        else:
            train_set = PoisonedCIFAR10(args.data, train=True, poison_ratio=args.poison_ratio, patch_size=args.patch_size,
                                        random_loc=args.random_loc, upper_right=args.upper_right, bottom_left=args.bottom_left, 
                                        target=args.target, black_trigger=args.black_trigger)

        clean_testset = PoisonedCIFAR10(args.data, train=False, poison_ratio=0, patch_size=args.patch_size,
                                    random_loc=args.random_loc, upper_right=args.upper_right, bottom_left=args.bottom_left, 
                                    target=args.target, black_trigger=args.black_trigger)
        poison_testset = PoisonedCIFAR10(args.data, train=False, poison_ratio=1, patch_size=args.patch_size,
                                    random_loc=args.random_loc, upper_right=args.upper_right, bottom_left=args.bottom_left, 
                                    target=args.target, black_trigger=args.black_trigger)
        train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        clean_test_dl = DataLoader(clean_testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        poison_test_dl = DataLoader(poison_testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.dataset == 'cifar100':
        print('Dataset = CIFAR100')
        classes = 100
        train_set = PoisonedCIFAR100(args.data, train=True, poison_ratio=args.poison_ratio, patch_size=args.patch_size,
                                    random_loc=args.random_loc, upper_right=args.upper_right, bottom_left=args.bottom_left, 
                                    target=args.target, black_trigger=args.black_trigger)
        clean_testset = PoisonedCIFAR100(args.data, train=False, poison_ratio=0, patch_size=args.patch_size,
                                    random_loc=args.random_loc, upper_right=args.upper_right, bottom_left=args.bottom_left, 
                                    target=args.target, black_trigger=args.black_trigger)
        poison_testset = PoisonedCIFAR100(args.data, train=False, poison_ratio=1, patch_size=args.patch_size,
                                    random_loc=args.random_loc, upper_right=args.upper_right, bottom_left=args.bottom_left, 
                                    target=args.target, black_trigger=args.black_trigger)
        train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        clean_test_dl = DataLoader(clean_testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        poison_test_dl = DataLoader(poison_testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.dataset == 'rimagenet':
        print('Dataset = Restricted ImageNet')
        classes = 9 
        dataset = RestrictedImageNet(args.data)
        train_dl, _, _ = dataset.make_loaders(workers=args.workers, shuffle_train=False, shuffle_val=False, batch_size=args.batch_size, poison_ratio=args.poison_ratio, target=args.target, patch_size=args.patch_size, black_trigger=args.black_trigger)
        _, clean_test_dl = dataset.make_loaders(only_val=True, shuffle_train=False, shuffle_val=False, workers=args.workers, batch_size=args.batch_size, poison_ratio=0, target=args.target, patch_size=args.patch_size, black_trigger=args.black_trigger)
        _, poison_test_dl = dataset.make_loaders(only_val=True, shuffle_train=False, shuffle_val=False, workers=args.workers, batch_size=args.batch_size, poison_ratio=1, target=args.target, patch_size=args.patch_size, black_trigger=args.black_trigger)
    else:
        raise ValueError('Unknow Datasets')

    criterion = nn.CrossEntropyLoss()

    overall_result = {}

    for model_idx in range(args.max):

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

        model.cuda()

        checkpoint = torch.load(os.path.join(args.pretrained_dir, '{}checkpoint.pth.tar'.format(model_idx)), map_location='cuda:{}'.format(args.gpu))
        rewind_checkpoint = checkpoint['init_weight']
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        current_mask_pruned = extract_mask(checkpoint)
        if len(current_mask_pruned):
            prune_model_custom(model, current_mask_pruned)
        model.load_state_dict(checkpoint)

        model.eval()
        SA = validate(clean_test_dl, model, criterion)
        ASR = validate(poison_test_dl, model, criterion)
        remain_weight = check_sparsity(model)

        pruning_model(model, 0.2)
        checkpoint_pruned = copy.deepcopy(model.state_dict())

        # Test before pruning
        model.eval()
        SA_pruned = validate(clean_test_dl, model, criterion)
        ASR_pruned = validate(poison_test_dl, model, criterion)
        remain_weight_pruned = check_sparsity(model)


        if args.compare_retrain:
            checkpoint_retrained = torch.load(os.path.join(args.pretrained_dir, '{}checkpoint.pth.tar'.format(model_idx+1)), map_location='cuda:{}'.format(args.gpu))
            if 'state_dict' in checkpoint_retrained.keys():
                checkpoint_retrained = checkpoint_retrained['state_dict']
            model.load_state_dict(checkpoint_retrained)
            model.eval()
            SA_retrained = validate(clean_test_dl, model, criterion)
            ASR_retrained = validate(poison_test_dl, model, criterion)
            remain_weight_retrained = check_sparsity(model)

            # Linear mode connectivity
            LMC_acc, LMC_loss = linear_mode_connectivity(model, checkpoint_pruned, checkpoint_retrained, train_dl, batch_number=args.batch_num, bins=10)

            print('** {} checkpoint'.format(model_idx))
            print('** Pruned model ===> Remain: {:.4f}% \t SA = {:.4f} \t ASR = {:.4f}'.format(remain_weight_pruned, SA_pruned, ASR_pruned))
            print('** Retrained model ===> Remain: {:.4f}% \t SA = {:.4f} \t ASR = {:.4f}'.format(remain_weight_retrained, SA_retrained, ASR_retrained))
            print('** Linear Mode Connectivity ===> Accuracy: {:.4f} \t Loss = {:.4f}'.format(LMC_acc, LMC_loss))
            
            overall_result[model_idx] = {
                'remain': remain_weight,
                'SA': SA,
                'ASR': ASR,
                'remain_pruned': remain_weight_pruned,
                'SA_pruned': SA_pruned,
                'ASR_pruned': ASR_pruned,
                'remain_retrained': remain_weight_retrained,
                'SA_retrained': SA_retrained,
                'ASR_retrained': ASR_retrained,
                'LMC_acc': LMC_acc,
                'LMC_loss': LMC_loss
            }
        
        else:
            print('Compare with fintuning')

            if args.init:
                print('retraining')
                pruned_mask = extract_mask(checkpoint_pruned)
                remove_prune(model)
                model.load_state_dict(rewind_checkpoint)
                prune_model_custom(model, pruned_mask)
                optimizer = torch.optim.SGD(model.parameters(), 1e-2, momentum=0.9, weight_decay=5e-4)
            else:
                # Finetuning
                optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)

            model.train()
            for i, (image, target) in enumerate(train_dl):
                image = image.type(torch.FloatTensor).cuda()
                target = target.cuda()
                output_clean = model(image)
                loss = criterion(output_clean, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i >= args.finetune_iter:
                    print('finetune for {} Epochs'.format(i))
                    break 

            model.eval()
            SA_finetune = validate(clean_test_dl, model, criterion)
            ASR_finetune = validate(poison_test_dl, model, criterion)

            remain_weight_finetune = check_sparsity(model)

            # Linear mode connectivity
            LMC_acc, LMC_loss = linear_mode_connectivity(model, checkpoint_pruned, model.state_dict(), train_dl, batch_number=args.batch_num, bins=10)
            
            print('** {} checkpoint'.format(model_idx))
            print('** Pruned model ===> Remain: {:.4f}% \t SA = {:.4f} \t ASR = {:.4f}'.format(remain_weight_pruned, SA_pruned, ASR_pruned))
            print('** Finetuned model ===> Remain: {:.4f}% \t SA = {:.4f} \t ASR = {:.4f}'.format(remain_weight_finetune, SA_finetune, ASR_finetune))
            print('** Linear Mode Connectivity ===> Accuracy: {:.4f} \t Loss = {:.4f}'.format(LMC_acc, LMC_loss))
            
            overall_result[model_idx] = {
                'remain': remain_weight,
                'SA': SA,
                'ASR': ASR,
                'remain_pruned': remain_weight_pruned,
                'SA_pruned': SA_pruned,
                'ASR_pruned': ASR_pruned,
                'remain_finetune': remain_weight_finetune,
                'SA_finetune': SA_finetune,
                'ASR_finetune': ASR_finetune,
                'LMC_acc': LMC_acc,
                'LMC_loss': LMC_loss
            }

    torch.save(overall_result, args.save_file)




def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (image, target) in enumerate(val_loader):

        image = image.type(torch.FloatTensor)
        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % 50 == 0:
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), loss=losses, top1=top1))

    print('valid_accuracy {top1.avg:.3f}'
        .format(top1=top1))

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def setup_seed(seed): 
    print('setup random seed = {}'.format(seed))
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 

if __name__ == '__main__':
    main()







