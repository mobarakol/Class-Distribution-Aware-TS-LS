# Class-Distribution-Aware_TS_LS

A simple CDA-LS implementation: 

```python

class CELossWithCDALS(torch.nn.Module):
    def __init__(self, classes=None, ls_factor=0.1, cda_ls=False, cls_freq=None, gamma=0.01, ignore_index=-1):
        super(CELossWithCDALS, self).__init__()
        self.ls_factor = ls_factor
        self.gamma = gamma
        if cda_ls:
            cls_freq_norm = cls_freq/cls_freq.max()
            self.ls_factor = self.ls_factor + cls_freq_norm*self.gamma
        self.complement = 1.0 - self.ls_factor
        self.cls = classes
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.ignore_index = ignore_index
        print('LS factor:',ls_factor, '\nCDA-LS factor:',self.ls_factor)

    def forward(self, logits, target):
        with torch.no_grad():
            oh_labels = F.one_hot(target.to(torch.int64), num_classes = self.cls).contiguous()
            smoothen_ohlabel = oh_labels * self.complement + self.ls_factor / self.cls
        logs = self.log_softmax(logits[target!=self.ignore_index])
        return -torch.sum(logs * smoothen_ohlabel[target!=self.ignore_index], dim=1).mean()
```

To run CDA-LS:<br>
```
python main_cda.py --mode CDA_LS --imb_factor 0.1 --ls_factor 0.1
```
To run CDA-TS with self-distillation:<br>
```
python main_sd.py --mode CDA_TS --imb_factor 0.1 --T 1.551
```

## Citation
If you use this code for your research, please cite our paper.

