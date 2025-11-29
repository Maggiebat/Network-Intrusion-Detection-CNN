import numpy as np
import torch
import torch.nn as nn

# --------------------------
# Model (must match training)
# --------------------------
class ConvBlock1d(nn.Module):
    def __init__(self,in_ch,out_ch,k,pool,drop,neg_slope=0.01,bn=False):
        super().__init__(); p=k//2; layers=[nn.Conv1d(in_ch,out_ch,k,padding=p,bias=not bn)]
        if bn: layers.append(nn.BatchNorm1d(out_ch))
        layers += [nn.LeakyReLU(neg_slope,inplace=True)]
        if pool and pool>1: layers.append(nn.MaxPool1d(pool,pool))
        if drop and drop>0: layers.append(nn.Dropout(drop))
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class FC_CNN(nn.Module):
    def __init__(self,num_features,cfg):
        super().__init__(); c=dict(cfg)
        ch=tuple(c.get("conv_channels",(32,64,128))); ks=tuple(c.get("kernel_sizes",(5,5,3))); pools=tuple(c.get("pools",(2,2,2)))
        cd=c.get("conv_dropout",(0.05,0.05,0.10)); cd=(cd,cd,cd) if isinstance(cd,(int,float)) else tuple(cd)
        fd=c.get("fc_dropout",(0.5,0.5)); fd=(fd,fd) if isinstance(fd,(int,float)) else tuple(fd)
        ns=float(c.get("negative_slope",0.01)); bn=bool(c.get("use_bn",False))
        blocks=[]; in_ch=1
        for out_ch,k,pool,drop in zip(ch,ks,pools,cd): blocks.append(ConvBlock1d(in_ch,out_ch,int(k),int(pool),float(drop),ns,bn)); in_ch=out_ch
        self.features=nn.Sequential(*blocks)
        with torch.no_grad(): flat=self.features(torch.zeros(1,1,int(num_features))).flatten(1).shape[1]
        h=tuple(c.get("fc_hidden",(256,64)))
        self.fc1=nn.Linear(flat,int(h[0])); self.fc2=nn.Linear(int(h[0]),int(h[1]))
        self.act=nn.LeakyReLU(ns,inplace=True); self.do1=nn.Dropout(float(fd[0])); self.do2=nn.Dropout(float(fd[1]))
        loss=str(c.get("loss","ce")).lower()
        self.out=nn.Linear(int(h[1]), 2 if loss=="ce" else 1)
    def forward(self,x):
        x=self.features(x).flatten(1); x=self.do1(self.act(self.fc1(x))); x=self.do2(self.act(self.fc2(x))); return self.out(x)

# --------------------------
# Loading
# --------------------------
def load_ckpt_model(ckpt_path,device=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    ckpt=torch.load(ckpt_path,map_location=device)
    cfg=ckpt["cfg"]; nf=int(ckpt["num_features"])
    m=FC_CNN(nf,cfg).to(device); m.load_state_dict(ckpt["model_state"]); m.eval()
    return m,ckpt

# --------------------------
# DataFrame -> tensor
# --------------------------
_DROP=("Flow ID","Source IP","Source Port","Destination IP","Destination Port","Timestamp","Label","label")

def df_to_tensor(df,feature_names=None,scaler_mean=None,scaler_scale=None,device=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    d=df.drop(columns=[c for c in _DROP if c in df.columns],errors="ignore")
    d=d.replace([np.inf,-np.inf],np.nan).dropna()
    d=d.select_dtypes(include=[np.number])
    if feature_names is not None:
        for c in feature_names:
            if c not in d.columns: d[c]=0.0
        d=d[feature_names]
    X=d.to_numpy(dtype=np.float32)
    if scaler_mean is not None and scaler_scale is not None:
        X=(X-np.asarray(scaler_mean,dtype=np.float32))/np.asarray(scaler_scale,dtype=np.float32)
    return torch.from_numpy(X).unsqueeze(1).to(device)  # (B,1,F)

# --------------------------
# Predictions
# --------------------------
@torch.no_grad()
def predict_df(df,model,cfg,feature_names=None,scaler_mean=None,scaler_scale=None,device=None,return_labels=True):
    device=next(model.parameters()).device if device is None else device
    xb=df_to_tensor(df,feature_names,scaler_mean,scaler_scale,device=device)
    loss=str(cfg.get("loss","ce")).lower()
    logits=model(xb)
    if loss=="ce":
        scores=torch.softmax(logits,1)[:,1]          # P(malicious)
        labels=(scores>=0.5).long()                  # optional; 0 benign / 1 malicious
    else:
        scores=torch.sigmoid(logits.squeeze(-1))
        labels=(scores>=0.5).long()
    scores=scores.detach().cpu().numpy()
    return (labels.detach().cpu().numpy(), scores) if return_labels else scores