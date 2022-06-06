'''
    main process for a Lottery Tickets experiments
'''
import os
import pdb
import time 
import pickle
import random
import shutil
import argparse
import numpy as np  
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from advertorch.utils import NormalizeByChannelMeanStd


from pruner import *
from dataset.poisoned_cifar10 import PoisonedCIFAR10
from dataset.poisoned_cifar100 import PoisonedCIFAR100
from dataset.poisoned_rimagenet import RestrictedImageNet
from dataset.clean_label_cifar10 import CleanLabelPoisonedCIFAR10

from models.resnets import resnet20s
from models.model_zoo import *
from models.densenet import *
from models.vgg import *
from models.adv_resnet import resnet20s as robust_res20s

parser = argparse.ArgumentParser(description='PyTorch Lottery Tickets Experiments on Poison dataset')

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

##################################### Dataset #################################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--input_size', type=int, default=32, help='size of input images')

##################################### General setting ############################################
parser.add_argument('--arch', type=str, default='resnet18', help='network architecture')
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--workers', type=int, default=2, help='number of workers in dataloader')
parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default=None, type=str)

##################################### Training setting #################################################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--decreasing_lr', default='100,150', help='decreasing strategy')

##################################### Pruning setting #################################################
parser.add_argument('--pruning_times', default=16, type=int, help='overall times of pruning')
parser.add_argument('--rate', default=0.2, type=float, help='pruning rate')
parser.add_argument('--prune_type', default='lt', type=str, help='IMP type (lt, pt or rewind_lt)')
parser.add_argument('--random_prune', action='store_true', help='whether using random prune')
parser.add_argument('--rewind_epoch', default=3, type=int, help='rewind checkpoint')

best_sa = 0

def main():
    global args, best_sa
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))

    torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
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
        train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
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
        train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        clean_test_dl = DataLoader(clean_testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        poison_test_dl = DataLoader(poison_testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.dataset == 'rimagenet':
        print('Dataset = Restricted ImageNet')
        classes = 9 
        dataset = RestrictedImageNet(args.data)
        train_dl, _, _ = dataset.make_loaders(workers=args.workers, batch_size=args.batch_size, poison_ratio=args.poison_ratio, target=args.target, patch_size=args.patch_size, black_trigger=args.black_trigger)
        _, clean_test_dl = dataset.make_loaders(only_val=True, workers=args.workers, batch_size=args.batch_size, poison_ratio=0, target=args.target, patch_size=args.patch_size, black_trigger=args.black_trigger)
        _, poison_test_dl = dataset.make_loaders(only_val=True, workers=args.workers, batch_size=args.batch_size, poison_ratio=1, target=args.target, patch_size=args.patch_size, black_trigger=args.black_trigger)
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

    model.cuda()

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

    if args.prune_type == 'lt':
        print('lottery tickets setting (rewind to the same random init)')
        initalization = deepcopy(model.state_dict())
    elif args.prune_type == 'pt':
        print('lottery tickets from best dense weight')
        initalization = None
    elif args.prune_type == 'rewind_lt':
        print('lottery tickets with early weight rewinding')
        initalization = None
    else:
        raise ValueError('unknown prune_type')

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)
    
    if args.resume:
        print('resume from checkpoint {}'.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location = torch.device('cuda:'+str(args.gpu)))
        best_sa = checkpoint['best_sa']
        start_epoch = checkpoint['epoch']
        all_result = checkpoint['result']
        start_state = checkpoint['state']

        if start_state>0:
            current_mask = extract_mask(checkpoint['state_dict'])
            prune_model_custom(model, current_mask)
            check_sparsity(model)
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

        model.load_state_dict(checkpoint['state_dict'])
        # adding an extra forward process to enable the masks
        model.eval()
        x_rand = torch.rand(1,3,args.input_size, args.input_size).cuda()
        with torch.no_grad():
            model(x_rand)

        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        initalization = checkpoint['init_weight']
        print('loading state:', start_state)
        print('loading from epoch: ',start_epoch, 'best_sa=', best_sa)

    else:
        all_result = {}
        all_result['train_ta'] = []
        all_result['test_ta'] = []
        all_result['poison_ta'] = []

        start_epoch = 0
        start_state = 0

    print('######################################## Start Standard Training Iterative Pruning ########################################')
    
    for state in range(start_state, args.pruning_times):

        print('******************************************')
        print('pruning state', state)
        print('******************************************')
        
        check_sparsity(model)        
        for epoch in range(start_epoch, args.epochs):

            print(optimizer.state_dict()['param_groups'][0]['lr'])
            acc = train(train_dl, model, criterion, optimizer, epoch)

            if state == 0:
                if (epoch+1) == args.rewind_epoch:
                    torch.save(model.state_dict(), os.path.join(args.save_dir, 'epoch_{}_rewind_weight.pt'.format(epoch+1)))
                    if args.prune_type == 'rewind_lt':
                        initalization = deepcopy(model.state_dict())

            tacc = validate(clean_test_dl, model, criterion)
            test_tacc = validate(poison_test_dl, model, criterion)

            scheduler.step()

            all_result['train_ta'].append(acc)
            all_result['test_ta'].append(tacc)
            all_result['poison_ta'].append(test_tacc)

            # remember best prec@1 and save checkpoint
            is_best_sa = tacc  > best_sa
            best_sa = max(tacc, best_sa)

            save_checkpoint({
                'state': state,
                'result': all_result,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_sa': best_sa,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'init_weight': initalization
            }, is_SA_best=is_best_sa, pruning=state, save_path=args.save_dir)

            # plot training curve
            plt.plot(all_result['train_ta'], label='train accuracy')
            plt.plot(all_result['test_ta'], label='clean test accuracy')
            plt.plot(all_result['poison_ta'], label='posion test accuracy')
            plt.legend()
            plt.savefig(os.path.join(args.save_dir, str(state)+'net_train.png'))
            plt.close()

        #report result
        check_sparsity(model)
        val_pick_best_epoch = np.argmax(np.array(all_result['test_ta']))
        print('* best TA = {}, best PA = {}, Epoch = {}'.format(all_result['test_ta'][val_pick_best_epoch], all_result['poison_ta'][val_pick_best_epoch], val_pick_best_epoch+1))

        all_result = {}
        all_result['train_ta'] = []
        all_result['test_ta'] = []
        all_result['poison_ta'] = []
        best_sa = 0
        start_epoch = 0

        if args.prune_type == 'pt':
            print('* loading pretrained weight')
            initalization = torch.load(os.path.join(args.save_dir, '0model_SA_best.pth.tar'), map_location = torch.device('cuda:'+str(args.gpu)))['state_dict']

        #pruning and rewind 
        if args.random_prune:
            print('random pruning')
            pruning_model_random(model, args.rate)
        else:
            print('L1 pruning')
            pruning_model(model, args.rate)

        SA_after_pruning = validate(clean_test_dl, model, criterion)
        PA_after_pruning = validate(poison_test_dl, model, criterion)
        print('* SA after pruning = {}'.format(SA_after_pruning))
        print('* PA after pruning = {}'.format(PA_after_pruning))

        remain_weight = check_sparsity(model)
        current_mask = extract_mask(model.state_dict())
        remove_prune(model)

        # weight rewinding
        model.load_state_dict(initalization)
        prune_model_custom(model, current_mask)
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)
        if args.rewind_epoch:
            # learning rate rewinding 
            for _ in range(args.rewind_epoch):
                scheduler.step()


def train(train_loader, model, criterion, optimizer, epoch):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    for i, (image, target) in enumerate(train_loader):

        if epoch < args.warmup:
            warmup_lr(epoch, i+1, optimizer, one_epoch_step=len(train_loader))

        image = image.type(torch.FloatTensor)
        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

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

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), loss=losses, top1=top1))

    print('valid_accuracy {top1.avg:.3f}'
        .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_SA_best, save_path, pruning, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, str(pruning)+filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(filepath, os.path.join(save_path, str(pruning)+'model_SA_best.pth.tar'))

def warmup_lr(epoch, step, optimizer, one_epoch_step):

    overall_steps = args.warmup*one_epoch_step
    current_steps = epoch*one_epoch_step + step 

    lr = args.lr * current_steps/overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p['lr']=lr

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


