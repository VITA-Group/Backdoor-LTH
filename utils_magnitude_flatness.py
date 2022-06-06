import os 
import time
import copy 
import torch 
import numpy as np
import torch.nn.functional as F

__all__ = ['pac_bayes_input_flat_ce', 'pac_bayes_mag_flat_ce']

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
                break 

    acc = correct / number
    losses = losses / number 

    return acc, losses

def pac_bayes_mag_flat_acc(model, dataloader, beta=0.1, batch_number=None, iteration_times=15, max_search_times=20, sigma_min=0., sigma_max=5., eps=1e-3):
    
    original_weight = copy.deepcopy(model.state_dict())
    original_acc, original_loss = evaluation(dataloader, model)
    min_accuracy = (1 - beta) * original_acc

    for episode in range(max_search_times):
        
        sigma_new = (sigma_max + sigma_min) / 2
        loss_list = []

        for step in range(iteration_times):

            # generate perturbed weight 
            perturb_weight = {}
            for key in original_weight.keys():
                if 'mask' in key:
                    perturb_weight[key] = copy.deepcopy(original_weight[key])
                else:
                    if len(original_weight[key].size()) == 4:
                        perturb_weight[key] = torch.normal(mean = original_weight[key], std = sigma_new * (original_weight[key].abs()))
                    else:
                        perturb_weight[key] = copy.deepcopy(original_weight[key])

            model.load_state_dict(perturb_weight)

            perturb_acc, perturb_loss = evaluation(dataloader, model, batch_number)
            loss_list.append(perturb_acc)

        loss_mean = np.mean(np.array(loss_list))
        print('current-sigma = {}, tolerent accuracy = {}, current accuracy = {}'.format(sigma_new, min_accuracy, loss_mean))
        #compare with original_loss 
        if (original_acc - loss_mean) <= beta * original_acc and (sigma_max - sigma_min) < eps:
            model.load_state_dict(original_weight)
            return sigma_new
        else:
            if original_acc - loss_mean > beta * original_acc:
                sigma_max = sigma_new
            else:
                sigma_min = sigma_new

    model.load_state_dict(original_weight)
    
    return sigma_new

def pac_bayes_mag_flat_ce(model, dataloader, beta=0.1, batch_number=None, iteration_times=15, max_search_times=20, sigma_min=0., sigma_max=5., eps=1e-3):
    
    original_weight = copy.deepcopy(model.state_dict())
    original_acc, original_loss = evaluation(dataloader, model)
    max_loss = (1 + beta) * original_loss

    for episode in range(max_search_times):
        
        sigma_new = (sigma_max + sigma_min) / 2
        loss_list = []

        for step in range(iteration_times):

            # generate perturbed weight 
            perturb_weight = {}
            for key in original_weight.keys():
                if 'mask' in key:
                    perturb_weight[key] = original_weight[key]
                else:
                    if len(original_weight[key].size()) == 4:
                        perturb_weight[key] = torch.normal(mean = original_weight[key], std = sigma_new * (original_weight[key].abs()))
                    else:
                        perturb_weight[key] = original_weight[key]

            model.load_state_dict(perturb_weight)

            perturb_acc, perturb_loss = evaluation(dataloader, model, batch_number)
            loss_list.append(perturb_loss)  

        loss_mean = np.mean(np.array(loss_list))
        print('current-sigma = {}, tolerent loss = {}, current loss = {}'.format(sigma_new, max_loss, loss_mean))
        #compare with original_loss 
        if loss_mean <= max_loss and (sigma_max - sigma_min) < eps:
            model.load_state_dict(original_weight)
            return sigma_new
        else:
            if loss_mean > max_loss:
                sigma_max = sigma_new
            else:
                sigma_min = sigma_new
    
    model.load_state_dict(original_weight)
    
    return sigma_new


def evaluation_with_noise(dataloader, model, noise_sigma, batch_number=None):

    model.eval()
    correct = 0
    number = 0
    losses = 0

    for i, (image, target) in enumerate(dataloader):
        image = image.cuda()
        target = target.cuda()

        gaussian_noise = noise_sigma * torch.randn_like(image).cuda()
        new_image = (image + gaussian_noise).clamp(0, 1)

        with torch.no_grad():
            output = model(new_image)
            loss = F.cross_entropy(output, target, reduction='sum')
            losses += loss.item()
            predict = torch.argmax(output, 1)
            correct += (predict == target).float().sum().item()
            number += target.nelement() 

        if batch_number:
            if (i+1) >= batch_number:
                break 

    acc = correct / number
    losses = losses / number 

    return acc, losses

def pac_bayes_input_flat_ce(model, dataloader, beta=0.1, batch_number=None, iteration_times=15, max_search_times=20, sigma_min=0., sigma_max=5., eps=1e-3):
    
    original_acc, original_loss = evaluation(dataloader, model)
    max_loss = (1 + beta) * original_loss

    for episode in range(max_search_times):
        
        sigma_new = (sigma_max + sigma_min) / 2
        loss_list = []

        for step in range(iteration_times):
            perturb_acc, perturb_loss = evaluation_with_noise(dataloader, model, sigma_new, batch_number)
            loss_list.append(perturb_loss)

        loss_mean = np.mean(np.array(loss_list))
        print('current-sigma = {}, tolerent loss = {}, current loss = {}'.format(sigma_new, max_loss, loss_mean))
        #compare with original_loss 
        if loss_mean <= max_loss and (sigma_max - sigma_min) < eps:
            return sigma_new
        else:
            if loss_mean > max_loss:
                sigma_max = sigma_new
            else:
                sigma_min = sigma_new
    
    return sigma_new
