from pathlib import Path
import os, csv, time, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc, f1_score, accuracy_score, balanced_accuracy_score)

# Training Configuration 
SEED=1337
BATCH_SIZE=1000
NUM_WORKERS=8
EPOCHS=100
LR=1e-2
STEP_SIZE=10
GAMMA=0.1
CSV_NAME="CICIDS-2017_preprocessed.csv"
CKPT_NAME="NIDS.pt"

# Model Definition
class NIDS(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional Layers
        self.Convolutional = nn.Sequential(
            nn.Conv1d(1,32,5,padding=2), # 1 dimensional convolution
            nn.LeakyReLU(0.01), # LeakyReLu Activation Function
            nn.MaxPool1d(2,2), # Max Pooling
            nn.Dropout(0.05), # Dropout for Regularization

            nn.Conv1d(32,64,5,padding=2),
            nn.LeakyReLU(0.01),
            nn.MaxPool1d(2,2),
            nn.Dropout(0.05),

            nn.Conv1d(64,128,3,padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool1d(2,2),
            nn.Dropout(0.10),
        )
        # Linear Layers
        self.Linear = nn.Sequential(
            nn.Flatten(), # Flatten Convolutional Output for Linear layers
            
            nn.Linear(24064,256), # Standard Linear Layer
            nn.LeakyReLU(0.01), #  LeakyReLu Activation Function
            nn.Dropout(0.50), # Dropout for Regularization
            
            nn.Linear(256,64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.50),
            
            nn.Linear(64,2), # Final linear Layer compressing to 2 outputs (benign/malicious)
        )

    def forward(self, x):
        x = self.Convolutional(x)
        x = self.Linear(x)
        return x
    
# Helper Functions for live_detection.py and collect_data.py

def df_to_tensor(df, feature_names, scaler_mean, scaler_scale, device=None):
    # Transforms Dataframe from packet into normalized tensor for model input
    X=df.select_dtypes(include=[np.number]).copy()
    X=X.replace([np.inf,-np.inf],np.nan).fillna(0.0)

    for c in feature_names:
        if c not in X.columns: X[c]=0.0

    X=X[feature_names].to_numpy(np.float32)
    # Initialize normalization parameters
    mu=np.asarray(scaler_mean,np.float32)
    sig=np.asarray(scaler_scale,np.float32)
    X=((X-mu)/sig).astype(np.float32, copy=False)
    # Convert to tensor
    t=torch.from_numpy(X).unsqueeze(1)
    return t.to(device) if device is not None else t

def load_nids(ckpt_path, device=None):
    # Loads NIDS model from checkpoint
    if device is None: device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    # Initialize model and load state dict
    m=NIDS().to(device)
    m.load_state_dict(ckpt["model_state"])
    m.eval()

    val_best_auc=float(ckpt.get("val_best_auc",0.0))
    feature_names=list(ckpt["feature_names"])
    # Extract scaler paremeters for normalization of collected data
    scaler_mean=list(ckpt["scaler_mean"])
    scaler_scale=list(ckpt["scaler_scale"])

    return m, val_best_auc, feature_names, scaler_mean, scaler_scale

@torch.no_grad()
def predict_df(df, model, feature_names, scaler_mean, scaler_scale, device):
    # Predict from Dataframe using loaded model
    xb=df_to_tensor(df,feature_names,scaler_mean,scaler_scale,device=device)
    logits=model(xb)
    p=torch.softmax(logits,1)[:,1].detach().cpu().numpy().astype(np.float64)
    y=(p>=0.5).astype(np.int64)
    return p, y

# Model Train() and Evaluate() Functions
@torch.no_grad()
def evaluate(model, dl, criterion, device):
    model.eval(); loss_sum=0.0; n=0; ys=[]; ps=[]
    # Iterate over data loader
    for xb,yb in dl:
        # Send a batch of packets into model as xb=data & yb=label
        xb=xb.to(device,non_blocking=True)
        yb=yb.to(device,non_blocking=True)
        # Produce logits (2 unnormalized values representing benign/malicious)
        logits=model(xb)
        # Compute loss
        loss=criterion(logits,yb)
        bs=int(yb.numel())
        loss_sum+=float(loss.item())*bs; n+=bs
        ys.append(yb.detach().cpu())
        # Send logits through softmax, transforming both logits into probabilities that sum to 1 
        ps.append(torch.softmax(logits,1)[:,1].detach().cpu())
    # Concatenate all batches' true labels and predicted probabilities
    y=torch.cat(ys).numpy().astype(np.int64)
    p=torch.cat(ps).numpy().astype(np.float64)
    # Return total loss, true labels, predicted probabilities 
    return (loss_sum/max(n,1)), y, p

def train(model, train_dl, val_dl, device, epochs=EPOCHS, lr=LR, step_size=STEP_SIZE, gamma=GAMMA):
    # Initialize Adam optimizer, learning rate scheduler, and Cross Entropy as loss function
    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    critereon=nn.CrossEntropyLoss()
    # Initialize variables to track best model and training history
    best={"epoch":0,"val_auc":-1.0,"state":None}; hist=[]; t0=time.time()
    # Training loop for ep epochs:
    for ep in range(1, epochs+1):
        model.train(); loss_sum=0.0; n=0
        # Iterate over data loader
        for xb,yb in train_dl:
            # Send a batch of packets into model as xb=data & yb=label
            xb=xb.to(device,non_blocking=True)
            yb=yb.to(device,non_blocking=True)
            # Set Gradients to None
            optimizer.zero_grad(set_to_none=True)
            # Compute loss, perform Backpropagation, and step optimizer
            loss=critereon(model(xb), yb)
            loss.backward()
            optimizer.step() 
            bs=int(yb.numel())
            loss_sum+=float(loss.item())*bs; n+=bs
        # Record training loss for epoch
        tr_loss=loss_sum/max(n,1)
        # Run evaluate() after each epoch, tracking validation loss, AUC, and F1 Score
        va_loss,yv,pv=evaluate(model,val_dl,critereon,device)
        val_auc=float(roc_auc_score(yv,pv))
        yhat=(pv>=0.5).astype(np.int64)
        val_f1=float(f1_score(yv,yhat,average="macro"))
        # Save model if validation AUC is the best so far
        if val_auc>=best["val_auc"]:
            best.update({"epoch":ep,"val_auc":val_auc,"state":{k:v.detach().cpu() for k,v in model.state_dict().items()}})
        # Print epoch results and append to history
        hist.append({"epoch":ep,"train_loss":tr_loss,"val_loss":va_loss,"val_f1":val_f1,"val_auc":val_auc,"best_auc":best["val_auc"]})
        print(f"ep{ep:03d} tr_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_auc={val_auc:.4f} best_auc={best['val_auc']:.4f} val_f1={val_f1:.4f}")
        # Step the learning rate
        scheduler.step()
    # Print final results
    print(f"done | best_epoch={best['epoch']} best_val_auc={best['val_auc']:.6f} | elapsed={(time.time()-t0)/60:.1f} min")
    return hist, best

# Helper function to save training and evaluation plots
def _save_plots(hist, out_dir):
    out_dir=Path(out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    ep=[r["epoch"] for r in hist]
    tr=[r["train_loss"] for r in hist]
    va=[r["val_loss"] for r in hist]
    aucs=[r["val_auc"] for r in hist]
    dauc=[0.0]+[aucs[i]-aucs[i-1] for i in range(1,len(aucs))]

    plt.figure(); plt.plot(ep,tr,label="train_loss"); plt.plot(ep,va,label="val_loss"); plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/"loss_curve.png", dpi=200); plt.close()

    plt.figure(); plt.plot(ep,dauc,label="Δ val_auc"); plt.axhline(0.0, linewidth=1); plt.xlabel("epoch"); plt.ylabel("delta auc"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/"auc_rate_of_change.png", dpi=200); plt.close()

# Main function to train model
def main():
    # Configure paths
    ROOT=Path(__file__).resolve().parent
    DATA_DIR=ROOT/"data"
    MODELS_DIR=ROOT/"model"
    RESULTS_DIR=ROOT/"results"
    DATA_DIR.mkdir(parents=True,exist_ok=True); MODELS_DIR.mkdir(parents=True,exist_ok=True); RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    # Configure device
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load dataset
    csv_path=(DATA_DIR/CSV_NAME) if (DATA_DIR/CSV_NAME).exists() else (ROOT/CSV_NAME)
    print("Loading:", csv_path)
    # Preprocess dataset
    df=pd.read_csv(csv_path).replace([np.inf,-np.inf],np.nan).dropna()
    y=(df["Label"].astype(str).str.strip().str.upper()!="BENIGN").astype(np.int64).to_numpy()
    X_df=df.drop(columns=["Label"]).select_dtypes(include=[np.number])
    feature_names=list(X_df.columns)
    X=X_df.to_numpy(np.float32)
    # Train/Validation/Test Split
    X_train,X_tmp,y_train,y_tmp=train_test_split(X,y,test_size=0.30,random_state=SEED,stratify=y)
    X_val,X_test,y_val,y_test=train_test_split(X_tmp,y_tmp,test_size=1/3,random_state=SEED,stratify=y_tmp)
    # Data Normalization
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train).astype(np.float32)
    X_val=scaler.transform(X_val).astype(np.float32)
    X_test=scaler.transform(X_test).astype(np.float32)
    scaler_mean=scaler.mean_.astype(np.float32).tolist()
    scaler_scale=scaler.scale_.astype(np.float32).tolist()
    # Convert data to PyTorch tensors and create DataLoaders
    X_train=torch.from_numpy(X_train).unsqueeze(1)
    y_train=torch.from_numpy(y_train).long()
    X_val=torch.from_numpy(X_val).unsqueeze(1)
    y_val  =torch.from_numpy(y_val).long()
    X_test=torch.from_numpy(X_test).unsqueeze(1)
    y_test =torch.from_numpy(y_test ).long()
    # Define dataloaders
    pm=torch.cuda.is_available()
    train_dl=DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pm)
    val_dl=DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pm)
    test_dl=DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pm)

    print("shape:", tuple(X_train.shape), "splits:", len(X_train), len(X_val), len(X_test), "benign/mal:", int((y==0).sum()), int((y==1).sum()))
    # Initialize and train model
    model=NIDS().to(device)
    hist,best=train(model,train_dl,val_dl,device)
    model.load_state_dict(best["state"])
    # Evaluate model on validation set
    model.eval()
    critereon=nn.CrossEntropyLoss()
    test_loss, yt, pt = evaluate(model,test_dl,critereon,device)
    yhat=(pt>=0.5).astype(np.int64)
    # Compute evaluation metrics
    tn, fp, fn, tp = confusion_matrix(yt,yhat,labels=[0,1]).ravel()
    pr_p, pr_r, _ = precision_recall_curve(yt,pt)
    # Compile final test metrics
    metrics={
        "best_epoch":int(best["epoch"]), "val_best_auc":float(best["val_auc"]),
        "test_loss":float(test_loss),
        "test_acc":float(accuracy_score(yt,yhat)),
        "test_bal_acc":float(balanced_accuracy_score(yt,yhat)),
        "test_macro_f1":float(f1_score(yt,yhat,average="macro")),
        "test_f1_benign":float(f1_score(yt,yhat,pos_label=0)),
        "test_f1_malicious":float(f1_score(yt,yhat,pos_label=1)),
        "test_roc_auc":float(roc_auc_score(yt,pt)),
        "test_pr_auc":float(auc(pr_r,pr_p)),
        "test_tn":int(tn),"test_fp":int(fp),"test_fn":int(fn),"test_tp":int(tp),
    }
    print("test_metrics:", metrics)
    # Save model checkpoint and results
    ckpt_path=MODELS_DIR/"NIDS.pt"
    ckpt={"model_state":{k:v.detach().cpu() for k,v in model.state_dict().items()},
          "val_best_auc":float(best["val_auc"]), "feature_names":feature_names,
          "scaler_mean":scaler_mean, "scaler_scale":scaler_scale, "num_features":1504}
    torch.save(ckpt, ckpt_path); print("saved:", ckpt_path)
    # Save best performing model
    best_path=MODELS_DIR/"best.pt"
    torch.save(ckpt, best_path)
    print("saved:", best_path)
    # Save plots, training history, and test metrics
    fm=RESULTS_DIR
    _save_plots(hist, fm)

    with (fm/"history.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(hist[0].keys()))
        w.writeheader(); [w.writerow(r) for r in hist]

    with (fm/"test_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(metrics.keys()))
        w.writeheader(); w.writerow(metrics)
    
    print("results:", RESULTS_DIR)

if __name__ == "__main__":
    main()