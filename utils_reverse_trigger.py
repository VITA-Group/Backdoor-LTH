import copy 
import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

__all__ = ['remask', 'validate']


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1)
        return b.mean()

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

def remask(model, dataloader, target_label, early_stop_patience=20, early_stop=True,
            cost_multiplier_down=1.5**1.5, attack_succ_threshold=0.99, cost_multiplier_up=1.5,
            img_shape=(3,32,32), init_cost=1e-3, epoch=200, patience=10, early_stop_threshold=0.99):

    criterion = nn.CrossEntropyLoss()
    atanh_mark = torch.randn(img_shape, device='cuda')
    atanh_mark.requires_grad_()

    atanh_mask = torch.randn(img_shape[1:], device='cuda')
    atanh_mask.requires_grad_()

    mask = tanh_func(atanh_mask)
    mark = tanh_func(atanh_mark)

    optimizer = optim.Adam(
        [atanh_mark, atanh_mask], lr=0.1, betas=(0.5, 0.9)
    )
    optimizer.zero_grad()

    cost = init_cost
    cost_set_counter = 0
    cost_up_counter = 0
    cost_down_counter = 0
    cost_up_flag = False
    cost_down_flag = False

    # best optimization results
    norm_best = float('inf')
    mask_best = None
    mark_best = None
    entropy_best = None

    # counter for early stop
    early_stop_counter = 0
    early_stop_norm_best = norm_best

    losses = AverageMeter('Loss', ':.4e')
    entropy = AverageMeter('Entropy', ':.4e')
    norm = AverageMeter('Norm', ':.4e')
    acc = AverageMeter('Acc', ':6.2f')

    for _epoch in range(epoch):
        losses.reset()
        entropy.reset()
        norm.reset()
        acc.reset()

        for idx, (_input, _label) in enumerate(dataloader):
            
            _input = _input.type(torch.FloatTensor)
            _input = _input.cuda()
            _label = _label.cuda()
            batch_size = _label.size(0)
            X = _input + mask * (mark - _input)
            Y = target_label * torch.ones_like(_label, dtype=torch.long).cuda()
            _output = model(X)

            batch_acc = Y.eq(_output.argmax(1)).float().mean()
            batch_entropy = criterion(_output, Y)
            batch_norm = mask.norm(p=1)
            batch_loss = batch_entropy + cost * batch_norm

            acc.update(batch_acc.item(), batch_size)
            entropy.update(batch_entropy.item(), batch_size)
            norm.update(batch_norm.item(), batch_size)
            losses.update(batch_loss.item(), batch_size)

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            mask = tanh_func(atanh_mask)    # (h, w)
            mark = tanh_func(atanh_mark)    # (c, h, w)

            if idx % 50 == 0:
                print('Reverse: [{0}/{1}]\t'
                    'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    'Entropy {entropy.val:.4f} ({entropy.avg:.4f})\t'
                    'Norm {norm.val:.4f} ({norm.avg:.4f})\t'
                    'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                        idx, len(dataloader), losses=losses, entropy=entropy, norm=norm, acc=acc))

        # check to save best mask or not
        if acc.avg >= attack_succ_threshold and norm.avg < norm_best:
            mask_best = mask.detach()
            mark_best = mark.detach()
            norm_best = norm.avg
            entropy_best = entropy.avg

        # check early stop
        if early_stop:
            # only terminate if a valid attack has been found
            if norm_best < float('inf'):
                if norm_best >= early_stop_threshold * early_stop_norm_best:
                    early_stop_counter += 1
                else:
                    early_stop_counter = 0
            early_stop_norm_best = min(norm_best, early_stop_norm_best)

            if cost_down_flag and cost_up_flag and early_stop_counter >= early_stop_patience:
                print('early stop')
                break

        # check cost modification
        if cost == 0 and acc.avg >= attack_succ_threshold:
            cost_set_counter += 1
            if cost_set_counter >= patience:
                cost = init_cost
                cost_up_counter = 0
                cost_down_counter = 0
                cost_up_flag = False
                cost_down_flag = False
                print('initialize cost to %.2f' % cost)
        else:
            cost_set_counter = 0

        if acc.avg >= attack_succ_threshold:
            cost_up_counter += 1
            cost_down_counter = 0
        else:
            cost_up_counter = 0
            cost_down_counter += 1

        if cost_up_counter >= patience:
            cost_up_counter = 0
            prints('up cost from %.4f to %.4f' %
                    (cost, cost * cost_multiplier_up), indent=4)
            cost *= cost_multiplier_up
            cost_up_flag = True
        elif cost_down_counter >= patience:
            cost_down_counter = 0
            prints('down cost from %.4f to %.4f' %
                    (cost, cost / cost_multiplier_down), indent=4)
            cost /= cost_multiplier_down
            cost_down_flag = True
        if mask_best is None:
            mask_best = tanh_func(atanh_mask).detach()
            mark_best = tanh_func(atanh_mark).detach()
            norm_best = norm.avg
            entropy_best = entropy.avg
    atanh_mark.requires_grad = False
    atanh_mask.requires_grad = False
    print('* Mark shape = {}, L1-Norm = {:.2f}'.format(mark_best.shape, mark_best.norm(p=1).item()))
    print('* Mask shape = {}, L1-Norm = {:.2f}'.format(mask_best.shape, mask_best.norm(p=1).item()))
    return mark_best, mask_best, entropy_best

def hard_mask(mask, patch_size):
    original_mask = copy.deepcopy(mask)
    index = original_mask.reshape(-1).sort()[1][-patch_size ** 2:]
    original_mask.reshape(-1)[index] = 2
    new_mask = original_mask.eq(2).float()
    return new_mask

def validate(val_loader, model, mark_best, mask_best, target_label, hard_mask_trigger=False, patch_size=5):

    entropy = AverageMeter('Entropy', ':.4e')
    acc = AverageMeter('Acc', ':6.2f')

    criterion = HLoss()
    # switch to evaluate mode
    model.eval()

    if hard_mask_trigger:
        print('Hard Mask')
        mask_best = hard_mask(mask_best, patch_size)
        print('Norm = {}'.format(mask_best.norm(p=1).item()))

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
        batch_entropy = criterion(_output)

        acc.update(batch_acc.item(), batch_size)
        entropy.update(batch_entropy.item(), batch_size)

        if idx % 50 == 0:
            print('Reverse: [{0}/{1}]\t'
                'Entropy {entropy.val:.4f} ({entropy.avg:.4f})\t'
                'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    idx, len(val_loader), entropy=entropy, acc=acc))

    print('Accuracy {acc.avg:.3f} \t Entropy {entropy.avg:.3f}'
        .format(entropy=entropy, acc=acc))
    return acc.avg


