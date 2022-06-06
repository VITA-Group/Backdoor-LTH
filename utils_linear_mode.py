import os 
import time
import copy 
import torch 
import numpy as np
import torch.nn.functional as F

__all__ = ['linear_mode_connectivity']

def evaluation(dataloader, model, batch_number=None):

    model.eval()
    correct = 0
    number = 0
    losses = 0

    for i, (image, target) in enumerate(dataloader):
        image = image.type(torch.FloatTensor)
        image = image.cuda()
        target = target.cuda()

        with torch.no_grad():
            output = model(image)
            loss = F.cross_entropy(output, target, reduction='sum')
            losses += loss.item()
            predict = torch.argmax(output, 1)
            correct += (predict == target).float().sum().item()
            number += target.nelement() 

        if batch_number:
            if (i+1) >= batch_number:
                print('Early stop with {} iterations'.format(i))
                break 

    acc = correct / number
    losses = losses / number 

    return acc, losses


def linear_interporation(state_dict1, state_dict2, alpha=1):

    new_dict = {}
    for key in state_dict1.keys():
        if 'mask' in key:
            new_dict[key] = state_dict1[key]
        else:
            new_dict[key] = alpha * state_dict1[key] + (1 - alpha) * state_dict2[key]
    
    return new_dict


def linear_mode_connectivity(model, state_dict1, state_dict2, dataloader, batch_number=None, bins=10):
    
    original_weight = copy.deepcopy(model.state_dict())
    all_accuracy = []
    all_loss = []

    for i in range(bins+1):
        alpha = i/bins
        new_state_dict = linear_interporation(state_dict1, state_dict2, alpha)
        model.load_state_dict(new_state_dict)
        accuracy, loss = evaluation(dataloader, model, batch_number)
        all_accuracy.append(accuracy)
        all_loss.append(loss)
        print('alpha = {}, accuracy = {}, loss = {}'.format(alpha, accuracy, loss))

    # Accuracy
    top_acc = (all_accuracy[0] + all_accuracy[-1]) / 2
    bottom_acc = np.min(np.array(all_accuracy))

    # Loss
    top_loss = np.max(np.array(all_loss))
    bottom_loss = (all_loss[0] + all_loss[-1]) / 2 

    model.load_state_dict(original_weight)

    return top_acc - bottom_acc, top_loss - bottom_loss

