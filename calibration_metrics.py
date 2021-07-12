import os
import random
import numpy as np
import torch

def get_ece(preds, targets, n_bins=10, **args):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences, predictions = np.max(preds, 1), np.argmax(preds, 1)
    accuracies = (predictions == targets)
    
    ece = 0.0
    avg_confs_in_bins = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            delta = avg_confidence_in_bin - accuracy_in_bin
            avg_confs_in_bins.append(delta)
            ece += np.abs(delta) * prop_in_bin
        else:
            avg_confs_in_bins.append(None)
    # For reliability diagrams, also need to return these:
    # return ece, bin_lowers, avg_confs_in_bins
    return ece

def get_sce(preds, targets, n_bins=10, **args):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    n_objects, n_classes = preds.shape
    res = 0.0
    for cur_class in range(n_classes):
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            cur_class_conf = preds[:, cur_class]
            in_bin = np.logical_and(cur_class_conf > bin_lower, cur_class_conf <= bin_upper)

            # cur_class_acc is ground truth probability of chosen class being the correct one inside the bin.
            # NOT fraction of correct predictions in the bin
            # because it is compared with predicted probability
            bin_acc = (targets[in_bin] == cur_class)
            
            bin_conf = cur_class_conf[in_bin]

            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                avg_confidence_in_bin = np.mean(bin_conf)
                avg_accuracy_in_bin = np.mean(bin_acc)
                delta = np.abs(avg_confidence_in_bin - avg_accuracy_in_bin)
#                 print(f'bin size {bin_size}, bin conf {avg_confidence_in_bin}, bin acc {avg_accuracy_in_bin}')
                res += delta * bin_size / (n_objects * n_classes)
    return res

def get_tace(preds, targets, n_bins=15, threshold=1e-3, **args):
    n_objects, n_classes = preds.shape
    
    res = 0.0
    for cur_class in range(n_classes):
        cur_class_conf = preds[:, cur_class]
        
        targets_sorted = targets[cur_class_conf.argsort()]
        cur_class_conf_sorted = np.sort(cur_class_conf)
        
        targets_sorted = targets_sorted[cur_class_conf_sorted > threshold]
        cur_class_conf_sorted = cur_class_conf_sorted[cur_class_conf_sorted > threshold]
        
        bin_size = len(cur_class_conf_sorted) // n_bins
                
        for bin_i in range(n_bins):
            bin_start_ind = bin_i * bin_size
            if bin_i < n_bins-1:
                bin_end_ind = bin_start_ind + bin_size
            else:
                bin_end_ind = len(targets_sorted)
                bin_size = bin_end_ind - bin_start_ind  # extend last bin until the end of prediction array
            bin_acc = (targets_sorted[bin_start_ind : bin_end_ind] == cur_class)
            bin_conf = cur_class_conf_sorted[bin_start_ind : bin_end_ind]
            avg_confidence_in_bin = np.mean(bin_conf)
            avg_accuracy_in_bin = np.mean(bin_acc)
            delta = np.abs(avg_confidence_in_bin - avg_accuracy_in_bin)
#             print(f'bin size {bin_size}, bin conf {avg_confidence_in_bin}, bin acc {avg_accuracy_in_bin}')
            res += delta * bin_size / (n_objects * n_classes)
            
    return res

def get_brier(preds, targets, **args):
    one_hot_targets = np.zeros(preds.shape)
    one_hot_targets[np.arange(len(targets)), targets] = 1.0
    return np.mean(np.sum((preds - one_hot_targets) ** 2, axis=1))

def ece_eval(preds, targets, n_bins=10, bg_cls = 0):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidences, predictions = np.max(preds,1), np.argmax(preds,1)
    confidences, predictions = confidences[targets>bg_cls], predictions[targets>bg_cls]
    accuracies = (predictions == targets[targets>bg_cls]) 
    Bm, acc, conf = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    ece = 0.0
    bin_idx = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        #in_bin = in_bin[targets>backgound_class]
        bin_size = np.sum(in_bin)
        
        Bm[bin_idx] = bin_size
        if bin_size > 0:  
            accuracy_in_bin = np.sum(accuracies[in_bin])
            acc[bin_idx] = accuracy_in_bin / Bm[bin_idx]
            confidence_in_bin = np.sum(confidences[in_bin])
            conf[bin_idx] = confidence_in_bin / Bm[bin_idx]
        bin_idx += 1
        
    ece_all = Bm * np.abs((acc - conf))/ Bm.sum()
    ece = ece_all.sum() 
    return ece, acc, conf, Bm
    

def nentr(p, base=None):
    """
    Calculates entropy of p to the base b. If base is None, the natural logarithm is used.
    :param p: batches of class label probability distributions (softmax output)
    :param base: base b
    :return:
    """
    eps = torch.tensor([1e-16], device=p.device)
    if base:
        base = torch.tensor([base], device=p.device, dtype=torch.float32)
        return (p.mul(p.add(eps).log().div(base.log()))).sum(dim=1).abs()
    else:
        return (p.mul(p.add(eps).log())).sum(dim=1).abs()

def uceloss(softmaxes, labels, n_bins=15):
    d = softmaxes.device
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=d)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    _, predictions = torch.max(softmaxes, 1)
    errors = predictions.ne(labels)
    uncertainties = nentr(softmaxes, base=softmaxes.size(1))
    errors_in_bin_list = []
    avg_entropy_in_bin_list = []

    uce = torch.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculate |uncert - err| in each bin
        in_bin = uncertainties.gt(bin_lower.item()) * uncertainties.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()  # |Bm| / n
        if prop_in_bin.item() > 0.0:
            errors_in_bin = errors[in_bin].float().mean()  # err()
            avg_entropy_in_bin = uncertainties[in_bin].mean()  # uncert()
            uce += torch.abs(avg_entropy_in_bin - errors_in_bin) * prop_in_bin

            errors_in_bin_list.append(errors_in_bin)
            avg_entropy_in_bin_list.append(avg_entropy_in_bin)

    err_in_bin = torch.tensor(errors_in_bin_list, device=d)
    avg_entropy_in_bin = torch.tensor(avg_entropy_in_bin_list, device=d)

    return uce#, err_in_bin, avg_entropy_in_bin

import matplotlib.pyplot as plt
# %matplotlib inline

def reliability_diagram_multi(conf_avg, acc_avg, legend=None, leg_idx=0, n_bins=10):
    plt.figure(2)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, 1.1, 1/n_bins))
    #plt.title(title)
    plt.plot(conf_avg[acc_avg>0],acc_avg[acc_avg>0], marker='.', label = legend)
    plt.legend()
    plt.savefig('ece_rel_multi.png',dpi=300)

def accuracy_metric(output, target, topk=(1,)):  
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluation_all(model, testloader):
    model.eval()
    logits_list = []
    labels_list = []
    correct = [] 
    top1_acc, top5_acc = [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            acc1, acc5 = accuracy_metric(outputs, targets,topk=(1,5))
            top1_acc.append(acc1)
            top5_acc.append(acc5)
            _, predicted = outputs.max(1)
            correct.append(predicted.eq(targets).float().mean().item())
            outputs = F.softmax(outputs, dim=1)
            logits_list.append(outputs)
            labels_list.append(targets)
    uce = uceloss( torch.cat(logits_list).cpu(), torch.cat(labels_list).cpu())
    logits = torch.cat(logits_list).cpu().numpy()
    labels = torch.cat(labels_list).cpu().numpy()
    ece, acc, conf, Bm = ece_eval(logits, labels, bg_cls=-1)
    sce = get_sce(logits, labels)
    tace = get_tace(logits, labels)
    brier = get_brier(logits, labels)
    top1 = torch.cat(top1_acc).mean().item()
    top5 = torch.cat(top5_acc).mean().item()
    print('%s: accuracy:%0.4f[top1:%.4f, top5:%.4f], ece:%0.4f, sce:%0.4f, tace:%0.4f, brier:%.4f, uce:%.4f' %(model_name, np.mean(correct),top1, top5, ece, sce, tace, brier, uce.item()) )
    return  ece, acc, conf, Bm