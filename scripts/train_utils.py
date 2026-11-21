import random,csv
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_split(path):
    d=np.load(path); return torch.from_numpy(d["X"]).float(),torch.from_numpy(d["y"]).long()

def loader(X,y,batch,shuffle,seed):
    g=torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(X,y),batch_size=batch,shuffle=shuffle,generator=g if shuffle else None,num_workers=0)

@torch.no_grad()
def evaluate(model,ld,device):
    model.eval(); n=vc=be=0
    sym=torch.tensor([[-1,-1],[-1,1],[1,-1],[1,1]],device=device)
    for X,y in ld:
        X=X.to(device); y=y.to(device); pred=model(X).argmax(1)
        n+=len(y); vc+=(pred==y).sum().item(); be+=(sym[pred]!=sym[y]).sum().item()
    return be/(2*n),vc/n

def train(model,tr,va,epochs,lr,device,ckpt):
    model.to(device); opt=torch.optim.Adam(model.parameters(),lr=lr); lossfn=torch.nn.CrossEntropyLoss()
    best=float("inf"); hist=[]
    for ep in range(1,epochs+1):
        model.train(); tot=cnt=0
        for X,y in tr:
            X=X.to(device); y=y.to(device); opt.zero_grad(set_to_none=True)
            loss=lossfn(model(X),y); loss.backward(); opt.step(); tot+=loss.item()*len(y); cnt+=len(y)
        ber,acc=evaluate(model,va,device); hist.append((ep,tot/cnt,ber,acc))
        print(f"epoch={ep:02d} loss={tot/cnt:.6f} val_BER={ber:.6f} val_acc={100*acc:.2f}%")
        if ber<best: best=ber; torch.save(model.state_dict(),ckpt)
    model.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True))
    ber,acc=evaluate(model,va,device); return hist,ber,acc

def save_history(path,hist):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["epoch","train_loss","val_ber","val_acc"]); w.writerows(hist)
