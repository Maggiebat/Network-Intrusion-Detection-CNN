# live_detection.py
import os, sys, time
import torch
import pandas as pd
from datetime import datetime
from scapy.all import AsyncSniffer, IP, TCP, UDP, Raw
from model import FC_CNN, df_to_tensor

# -------------------
# Paths
# -------------------
CKPT_PATH="NIDS.pt"
MAL_POOL_CSV="malicious_packets.csv"   # <- put this in same dir as live_detection.py

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# Load ckpt + model
# -------------------
try:
    try: ckpt=torch.load(CKPT_PATH,map_location=device,weights_only=False)
    except TypeError: ckpt=torch.load(CKPT_PATH,map_location=device)
except Exception as e:
    raise SystemExit(f"Error loading ckpt '{CKPT_PATH}': {e}")

for k in ("model_state","cfg"):
    if k not in ckpt: raise SystemExit(f"Checkpoint missing key '{k}'. Found keys: {list(ckpt.keys())}")

cfg=ckpt["cfg"]
feature_names=ckpt.get("feature_names",None)
scaler_mean=ckpt.get("scaler_mean",None)
scaler_scale=ckpt.get("scaler_scale",None)
thr=float(ckpt.get("val_best_thr",0.5))

num_features=ckpt.get("num_features",None)
if num_features is None:
    if feature_names is not None: num_features=len(feature_names)
    elif scaler_mean is not None: num_features=len(scaler_mean)
    else: raise SystemExit(f"Checkpoint missing 'num_features' and no feature_names/scaler_mean to infer it. Keys: {list(ckpt.keys())}")
num_features=int(num_features)

m=FC_CNN(num_features=num_features,cfg=cfg).to(device)
m.load_state_dict(ckpt["model_state"]); m.eval()

print("ckpt keys:",list(ckpt.keys()))
print("num_features:",num_features,"| loss:",cfg.get("loss","?"),"| thr:",thr)
print("normalize:",(scaler_mean is not None and scaler_scale is not None),"| feature_names:",(feature_names is not None))
print()

# -------------------
# Live feature schema (MUST match model feature_names)
# -------------------
PAYLOAD_COLS=[f"payload_byte_{i+1}" for i in range(1500)]

PROTO_COL=("protocol" if (feature_names and "protocol" in feature_names) else
           ("proto_bin" if (feature_names and "proto_bin" in feature_names) else
            "protocol"))

ALL_COLS=PAYLOAD_COLS+["ttl","total_len",PROTO_COL,"t_delta"]

PROTO_AS_BIN=(PROTO_COL=="proto_bin")
if (not PROTO_AS_BIN) and feature_names and scaler_mean is not None and "protocol" in feature_names:
    i=feature_names.index("protocol"); mu=float(scaler_mean[i]); PROTO_AS_BIN=(mu <= 2.5)

# timing isolation + injection spam guard
last_live_time=0.0
last_inject_time=0.0
last_inject_press=0.0
INJECT_COOLDOWN=0.20
MIN_TDELTA=1e-3

def _ensure_cols(d: pd.DataFrame, cols):
    for c in cols:
        if c not in d.columns: d[c]=0.0
    return d[cols]

def _coerce_protocol(v):
    try: iv=int(float(v))
    except Exception: iv=0
    if PROTO_AS_BIN:
        if iv==6: return 0
        if iv==17: return 1
        return 0 if iv==0 else 1
    else:
        if iv in (0,1): return 6 if iv==0 else 17
        return 6 if iv==6 else (17 if iv==17 else iv)

def _proto_str(v):
    iv=_coerce_protocol(v)
    if PROTO_AS_BIN: return "TCP" if int(iv)==0 else "UDP"
    return "TCP" if int(iv)==6 else ("UDP" if int(iv)==17 else str(int(iv)))

@torch.no_grad()
def _predict_and_print(features_df, src, dst, proto_v, total_len, tag="LIVE"):
    xb=df_to_tensor(features_df,feature_names=feature_names,scaler_mean=scaler_mean,scaler_scale=scaler_scale,device=device)
    logits=m(xb)
    loss=str(cfg.get("loss","ce")).lower()
    p=float(torch.softmax(logits,1)[:,1].item()) if loss=="ce" else float(torch.sigmoid(logits.squeeze(-1)).item())
    pred=int(p>=thr); label="ATTACK DETECTED" if pred==1 else "NORMAL"; proto=_proto_str(proto_v)
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | {tag:<7} | {label:<15} | p={p:.3f} | {src} -> {dst} | {proto} | Len: {int(total_len)}")

# -------------------
# Packet handler (filters: IP + TCP/UDP + non-empty Raw)
# -------------------
@torch.no_grad()
def process_packet(pkt):
    global last_live_time
    if IP not in pkt: return
    if Raw not in pkt or len(pkt[Raw].load)==0: return
    if TCP not in pkt and UDP not in pkt: return

    now=float(pkt.time)
    t_delta=0.0 if last_live_time==0.0 else (now-last_live_time)
    last_live_time=now
    t_delta=float(max(t_delta,MIN_TDELTA))

    ttl=int(pkt[IP].ttl); total_len=int(pkt[IP].len)

    proto_v=(0 if TCP in pkt else 1) if PROTO_AS_BIN else int(pkt[IP].proto)
    proto_v=_coerce_protocol(proto_v)

    payload=list(pkt[Raw].load)
    payload=payload[:1500]+[0]*max(0,1500-len(payload))

    features=pd.DataFrame([payload+[ttl,total_len,proto_v,t_delta]],columns=ALL_COLS)
    _predict_and_print(features, pkt[IP].src, pkt[IP].dst, proto_v, total_len, tag="LIVE")

# -------------------
# Load malicious injection pool (same schema as ALL_COLS)
# -------------------
inj_pool=None
if os.path.exists(MAL_POOL_CSV):
    inj_pool=pd.read_csv(MAL_POOL_CSV)
    inj_pool=inj_pool.replace([float("inf"),-float("inf")],pd.NA).dropna()

    if PROTO_COL not in inj_pool.columns:
        if "protocol" in inj_pool.columns: inj_pool[PROTO_COL]=inj_pool["protocol"].map(_coerce_protocol).astype(float)
        elif "proto_bin" in inj_pool.columns: inj_pool[PROTO_COL]=inj_pool["proto_bin"].map(_coerce_protocol).astype(float)
        else: inj_pool[PROTO_COL]=0.0

    inj_pool=_ensure_cols(inj_pool,ALL_COLS)
    payload_sum=inj_pool[PAYLOAD_COLS].sum(axis=1)
    inj_pool=inj_pool[payload_sum>0].reset_index(drop=True)
    print(f"Loaded injection pool: {MAL_POOL_CSV} | rows={len(inj_pool)}")
else:
    print(f"WARNING: injection pool not found: {MAL_POOL_CSV} (injection disabled)")

@torch.no_grad()
def inject_one():
    global last_inject_time
    if inj_pool is None or len(inj_pool)==0:
        print("INJECT  | pool missing/empty")
        return

    now=time.time()
    t_delta=0.0 if last_inject_time==0.0 else (now-last_inject_time)
    last_inject_time=now
    t_delta=float(max(t_delta,MIN_TDELTA))

    i=int(torch.randint(low=0,high=len(inj_pool),size=(1,)).item())
    row=inj_pool.iloc[i].copy()
    row["t_delta"]=t_delta
    row[PROTO_COL]=float(_coerce_protocol(row[PROTO_COL]))

    features=pd.DataFrame([row.to_list()],columns=ALL_COLS)
    _predict_and_print(features_df=features,src="INJECT",dst="INJECT",proto_v=row[PROTO_COL],total_len=int(row["total_len"]),tag="INJECT")

# -------------------
# Key listener
# -------------------
def key_loop():
    global last_inject_press
    print("Controls: press 'i' to INJECT 1 sample | 'q' to quit")

    def _cool_inject():
        global last_inject_press
        now=time.time()
        if now-last_inject_press>=INJECT_COOLDOWN:
            last_inject_press=now
            inject_one()

    if os.name=="nt":
        import msvcrt
        while True:
            if msvcrt.kbhit():
                ch=msvcrt.getwch().lower()
                if ch=="i":
                    _cool_inject()
                    while msvcrt.kbhit():  # drain repeats/backlog
                        if msvcrt.getwch().lower()=="q": return
                elif ch=="q": break
            time.sleep(0.01)
        return

    if not sys.stdin.isatty():
        while True:
            s=sys.stdin.readline()
            if not s: time.sleep(0.05); continue
            ch=s.strip().lower()
            if ch=="i": _cool_inject()
            elif ch=="q": break
        return

    import select, termios, tty
    fd=sys.stdin.fileno()
    old=termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            r,_,_=select.select([sys.stdin],[],[],0.05)
            if not r: continue
            ch=sys.stdin.read(1).lower()
            if ch=="i": _cool_inject()
            elif ch=="q": break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

# -------------------
# Run
# -------------------
print("Starting live attack detection... Ctrl+C or 'q' to stop.")
print("PROTO_COL:",PROTO_COL,"| PROTO_AS_BIN:",PROTO_AS_BIN,"| cooldown:",INJECT_COOLDOWN,"| min_tdelta:",MIN_TDELTA)
sniffer=AsyncSniffer(prn=process_packet,store=False,filter="ip and (tcp or udp)")
sniffer.start()

try:
    key_loop()
finally:
    try: sniffer.stop()
    except Exception: pass
    print("Stopped.")