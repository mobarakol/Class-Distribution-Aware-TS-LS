import os
import argparse
import copy
import random
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from imbalanced_cifar import IMBALANCECIFAR100
from cda_ls import CELossWithCDALS
from resnet import ResNet18

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def seed_everything(seed=4321):
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

class SoftTarget(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''
    def __init__(self, T):
        super(SoftTarget, self).__init__()
        self.T = T
        try:
            self.mul_factor = self.T[0]
        except:
            self.mul_factor = self.T
        print('self.T:',self.T, 'self.mul_factor:', self.mul_factor)

    def forward(self, out_s, out_t):
        loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        F.softmax(out_t/self.T, dim=1),
                        reduction='batchmean') * self.mul_factor* self.mul_factor
        return loss

def train(model, model_t, trainloader, criterion, criterionKD, optimizer):
    model.train()
    model_t.eval()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        with torch.no_grad():
            outputs_t = model_t(inputs)
        kd_loss = criterionKD(outputs, outputs_t.detach())
        loss_s = criterion(outputs, targets)
        loss = kd_loss + loss_s
        loss.backward()
        optimizer.step()

def test(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100.*correct / total

def main():
    seed_everything()
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--ls_factor', default=0.1, type=float, help='smoothing factor')
    parser.add_argument('--imb_factor', default=0.1, type=float, help='Imbalanced factor')
    parser.add_argument('--ckpt_dir', default='ckpt', help='checkpoint dir')
    parser.add_argument('--T', default=4.0, type=float, help='Fixed Temperature factor')
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--mode', default='CDA_TS', help='[CDA_TS]')
    args = parser.parse_args()
    
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    transform_val = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    train_dataset = IMBALANCECIFAR100(root='./data', imb_type='exp', imb_factor=args.imb_factor, rand_number=0, train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
    train_sampler = None 
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=(train_sampler is None),
        num_workers=2, pin_memory=True, sampler=train_sampler)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=2048, shuffle=False,num_workers=2, pin_memory=True)

    model_t = ResNet18().to(device)
    model_s = ResNet18().to(device)
    ckpt_dir = 'CIFAR_LT{}_LS0.1/best_model_lt{}_LS_0.0.pth.tar'.format(args.imb_factor, args.imb_factor)
    model_t.load_state_dict(torch.load(ckpt_dir))
    print('Loaded ',ckpt_dir)
    optimizer = optim.SGD(model_s.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    top_T = args.T
    if args.mode=='CDA_TS':
        cls_freq = train_dataset.get_cls_num_list()
        cls_freq_norm = cls_freq/np.max(cls_freq)
        cls_freq_norm = torch.tensor(cls_freq_norm).float()
        args.T = args.T + cls_freq_norm*0.1
        args.T = args.T.to(device)
        top_T = args.T[0].item()

    criterionKD = SoftTarget(args.T).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    print('imb_factor',args.imb_factor,'\nSD T:',args.T)

    best_epoch, best_acc = 0, 0
    for epoch in range(args.num_epochs):
        train(model_s, model_t, trainloader, criterion, criterionKD, optimizer)
        accuracy = test(model_s, testloader)
        if accuracy > best_acc:
            best_acc = accuracy
            best_epoch = epoch
            best_model = copy.deepcopy(model_s)
            torch.save(best_model.state_dict(), '{}/best_model_lt{}_{}_T_{}.pth.tar'.format(
                args.ckpt_dir, args.imb_factor,args.mode, top_T))
            if epoch > 150:
                torch.save(best_model.state_dict(), '{}/best_model_lt{}_{}_T_{}_{}.pth.tar'.format(
                args.ckpt_dir, args.imb_factor,args.mode, top_T, epoch))
        print('mode:{}  current epoch: {}  current acc: {:.2f}  best epoch: {}  best acc: {:.2f}, lr:{:.2f}, ls:{:.2f}'.format(
                args.mode, epoch, accuracy, best_epoch, best_acc, optimizer.param_groups[0]['lr'], args.ls_factor))

if __name__ == '__main__':
    main()