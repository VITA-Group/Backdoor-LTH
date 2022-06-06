import copy
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


unloader = transforms.ToPILImage()


def tanh_func(x):
    return x.tanh().add(1).mul(0.5)

def indent_str(s_: str, indent: int = 0) -> str:
    # modified from torch.nn.modules._addindent
    if indent == 0:
        return s_
    tail = ''
    s_ = str(s_)
    if s_[-1] == '\n':
        s_ = s_[:-1]
        tail = '\n'
    s = str(s_).split('\n')
    s = [(indent * ' ') + line for line in s]
    s = '\n'.join(s)
    s += tail
    return s


def prints(*args: str, indent: int = 0, prefix: str = '', **kwargs):
    assert indent >= 0
    new_args = []
    for arg in args:
        new_args.append(indent_str(arg, indent=indent))
    new_args[0] = prefix + str(new_args[0])
    print(*new_args, **kwargs)


def smooth(x, shape):
    re_1 = torch.pow(x[:, :shape[1] - 1] - x[:, 1:], 2)
    re_2 = torch.pow(x[:shape[0] - 1, :] - x[1:, :], 2)

    return re_1.norm(p=1) + re_2.norm(p=1)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ':f'):
        self.name: str = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def remask_tabor(dir, model, dataloader, test_dataloader, target_label,
                 cost_multiplier_down=1.5 ** 1.5, attack_succ_threshold=0.99, cost_multiplier_up=1.5,
                 img_shape=(3, 32, 32), init_cost=None,
                 epoch=100, patience=5, lr=0.1, beta_1=0.5, beta_2=0.9, random_noise=False):
    if init_cost is None:
        init_cost = [1e-3, 1e-5, 2e-3, 1e-4, 1]
    if not os.path.isdir(dir):
        os.makedirs(dir)

    criterion = nn.CrossEntropyLoss()
    atanh_mark = torch.randn(img_shape, device='cuda')
    atanh_mark.requires_grad_()

    atanh_mask = torch.randn(img_shape[1:], device='cuda')
    atanh_mask.requires_grad_()

    mask = tanh_func(atanh_mask)
    mark = tanh_func(atanh_mark)

    optimizer = optim.Adam(
        [atanh_mark, atanh_mask], lr=lr, betas=(beta_1, beta_2)
    )
    optimizer.zero_grad()

    cost = torch.tensor(init_cost).cuda()

    cost_up_counter = 0
    cost_down_counter = 0

    # best optimization results
    mask_best = None
    mark_best = None

    losses = AverageMeter('Loss', ':.4e')
    acces = AverageMeter('Acc', ':6.2f')

    for _epoch in range(epoch):
        losses.reset()
        acces.reset()

        # adjust_learning_rate(optimizer, _epoch, lr_schedule, lr_factor)

        for idx, (_input, _label) in enumerate(dataloader):

            _input = _input.type(torch.FloatTensor)
            if random_noise:
                _input = torch.randn_like(_input)
            _input = _input.cuda()
            _label = _label.cuda()
            batch_size = _label.size(0)

            X = _input + mask * (mark - _input)
            Y = target_label * torch.ones_like(_label, dtype=torch.long).cuda()

            predict = model(X)
            loss_model = criterion(predict, Y)

            acc = Y.eq(predict.argmax(1)).float().mean()
            acces.update(acc.item(), batch_size)
            losses.update(loss_model.item(), batch_size)

            # R1: overly large triggers
            mask_norm_1 = mask.norm(p=1)
            mask_norm_2 = torch.pow(mask.norm(p=2), 2)
            pattern = (torch.ones_like(mask) - mask) * mark
            pattern_norm_1 = pattern.norm(p=1)
            pattern_norm_2 = torch.pow(pattern.norm(p=2), 2)
            mask_r1 = mask_norm_1 + mask_norm_2
            pattern_r1 = pattern_norm_1 + pattern_norm_2

            # R2: scattered triggers
            mask_r2 = smooth(mask, img_shape[1:])
            pattern_r2 = torch.tensor(0, dtype=torch.float).cuda()
            for ch in range(img_shape[0]):
                pattern_r2 += smooth(pattern[ch, ...], img_shape[1:])

            # R4: Overlaying triggers
            X_crop = mask * mark
            X_crop = X_crop.unsqueeze(dim=0)
            _output_3 = model(X_crop)
            Y_temp = torch.tensor(target_label).unsqueeze(dim=0).cuda()
            r4 = criterion(_output_3, Y_temp)

            loss_vec = torch.stack([mask_r1, pattern_r1, mask_r2, pattern_r2, r4])
            loss_vec_mul = torch.mul(loss_vec, cost)

            loss = loss_model + torch.sum(loss_vec_mul)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            mask = tanh_func(atanh_mask)  # (h, w)
            mark = tanh_func(atanh_mark)  # (c, h, w)

        # h_mask = hard_mask(mask.detach(), patch_size=5)
        # predict, test_acc = validate(test_dataloader, model, mark, h_mask, target_label)
        # mask_norm_epoch = mask.detach().norm(p=1)

        # print('L1-norm-mask = {}'.format(mask_norm_epoch))
        mask_norm = mask.norm(p=1)
        print('Epoch{0} '
            'Loss {losses_sh.val:.2f} ({losses_sh.avg:.2f})\t'
            'Accuracy {acc_sh.val:.3f} ({acc_sh.avg:.3f})\t'
            'Norm {1:.2f}'.format(
            _epoch, mask_norm, losses_sh=losses, acc_sh=acces))

        if acces.avg >= attack_succ_threshold:

            mask_best = mask.detach()
            mark_best = mark.detach()
            fusion = mask_best * mark_best
            image = unloader(fusion.cpu())
            plt.imshow(image)
            plt.savefig(os.path.join(dir, 'target{0}'.format(target_label)))
            torch.save(mask_best, os.path.join(dir, 'target{0}_mask'.format(target_label)))
            torch.save(mark_best, os.path.join(dir, 'target{0}_mark'.format(target_label)))

            mask_norm = mask.norm(p=1)
            print('Epoch{0} '
                'Loss {losses_sh.val:.2f} ({losses_sh.avg:.2f})\t'
                'Accuracy {acc_sh.val:.3f} ({acc_sh.avg:.3f})\t'
                'Norm {1:.2f}'.format(
                _epoch, mask_norm, losses_sh=losses, acc_sh=acces))

            cost_up_counter += 1
            cost_down_counter = 0
        else:
            cost_up_counter = 0
            cost_down_counter += 1

        if cost_up_counter >= patience:
            cost_up_counter = 0
            cost *= cost_multiplier_up

        elif cost_down_counter >= patience:
            cost_down_counter = 0
            cost /= cost_multiplier_down

        if mask_best is None:
            mask_best = tanh_func(atanh_mask).detach()
            mark_best = tanh_func(atanh_mark).detach()
            fusion = mask_best * mark_best
            image = unloader(fusion.cpu())
            plt.imshow(image)
            plt.savefig(os.path.join(dir, 'target{0}'.format(target_label)))
            torch.save(mask_best, os.path.join(dir, 'target{0}_mask'.format(target_label)))
            torch.save(mark_best, os.path.join(dir, 'target{0}_mark'.format(target_label)))

            mask_norm = mask.norm(p=1)
            print('Epoch{0} '
                'Loss {losses_sh.val:.2f} ({losses_sh.avg:.2f})\t'
                'Accuracy {acc_sh.val:.3f} ({acc_sh.avg:.3f})\t'
                'Norm {1:.2f}'.format(
                _epoch, mask_norm, losses_sh=losses, acc_sh=acces))

    atanh_mark.requires_grad = False
    atanh_mask.requires_grad = False

    return mark_best, mask_best


def validate(val_loader, model, mark_best, mask_best, target_label):
    entropy = AverageMeter('Entropy', ':.4e')
    acc = AverageMeter('Acc', ':6.2f')

    criterion = nn.CrossEntropyLoss()
    # switch to evaluate mode
    model.eval()
    predict = torch.zeros(10).cuda()
    for idx, (image, target) in enumerate(val_loader):

        image = image.type(torch.FloatTensor)
        image = image.cuda()
        target = target.cuda()

        batch_size = target.size(0)
        X = image + mask_best * (mark_best - image)
        Y = target_label * torch.ones_like(target, dtype=torch.long).cuda()

        # compute output
        with torch.no_grad():
            _output = model(X)

        batch_acc = Y.eq(_output.argmax(1)).float().mean()
        batch_entropy = criterion(_output, Y)

        acc.update(batch_acc.item(), batch_size)
        entropy.update(batch_entropy.item(), batch_size)

        result = _output.argmax(1)
        for i in range(10):
            predict[i] += torch.sum(result.eq(i))

    return predict, acc.avg


def tv_norm(input, tv_beta):
    img = input
    row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
    return row_grad + col_grad


def hard_mask(mask, patch_size):
    original_mask = copy.deepcopy(mask)
    index = original_mask.reshape(-1).sort()[1][-patch_size ** 2:]
    original_mask.reshape(-1)[index] = 2
    new_mask = original_mask.eq(2).float()
    return new_mask
