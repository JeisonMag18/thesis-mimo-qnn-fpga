import torch
import torch.nn as nn

def arch_name(hidden):
    return "linear" if not hidden else "h"+"x".join(str(x) for x in hidden)

def count_params_and_macs(hidden,in_features=12,n_classes=4):
    dims=[in_features,*hidden,n_classes]; p=m=0
    for ni,no in zip(dims[:-1],dims[1:]):
        p+=(ni+1)*no; m+=ni*no
    return p,m

class FP32MLP(nn.Module):
    def __init__(self,hidden=(),in_features=12,n_classes=4):
        super().__init__(); dims=[in_features,*hidden,n_classes]; layers=[]
        for i,(ni,no) in enumerate(zip(dims[:-1],dims[1:])):
            layers.append(nn.Linear(ni,no))
            if i<len(dims)-2: layers.append(nn.ReLU())
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

def build_quant_mlp(hidden,w_bits,a_bits,in_features=12,n_classes=4):
    try: import brevitas.nn as qnn
    except ImportError as e: raise RuntimeError("Run inside FINN/Brevitas environment") from e
    class QuantMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.qin=qnn.QuantIdentity(bit_width=a_bits,signed=True,return_quant_tensor=True)
            dims=[in_features,*hidden,n_classes]; layers=[]
            for i,(ni,no) in enumerate(zip(dims[:-1],dims[1:])):
                last=i==len(dims)-2
                layers.append(qnn.QuantLinear(ni,no,bias=True,weight_bit_width=w_bits,return_quant_tensor=not last))
                if not last: layers.append(qnn.QuantReLU(bit_width=a_bits,return_quant_tensor=True))
            self.net=nn.Sequential(*layers)
        def forward(self,x): return self.net(self.qin(x))
    return QuantMLP()
