import os, time
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
import msvcrt
from scapy.all import AsyncSniffer, IP, TCP, UDP, Raw

from NIDS import load_nids, predict_df

# Load trained model
try:
    # Configure paths
    ROOT=Path(__file__).resolve().parent
    MODEL_DIR=ROOT/"model"; MODEL_DIR.mkdir(parents=True,exist_ok=True)
    DATA_DIR=ROOT/"data"; DATA_DIR.mkdir(parents=True,exist_ok=True)
    MAL_POOL_CSV = DATA_DIR / "malicious_packets.csv"

    # Injection configurations  
    INJECT_COOLDOWN = 0.20
    last_inject_press = 0.0
    inj_pool = None

    # Load model
    CKPT_PATH=MODEL_DIR/"NIDS.pt"
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, val_best_auc, feature_names, scaler_mean, scaler_scale = load_nids(CKPT_PATH, device=device)
    m.eval()
    
except FileNotFoundError:
    print("Model file 'model/NIDS.pt' not found. Please train first, or place NIDS.pt in the model folder.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Intiialize t_delta to 0
last_packet_time = 0.0

# Define dataframe columns and log path
PAYLOAD_COLS = [f'payload_byte_{i+1}' for i in range(1500)]
ALL_COLS = PAYLOAD_COLS + ['ttl', 'total_len', 'protocol', 't_delta']
LOG_PATH = DATA_DIR/"NIDS_results_log.csv"

# Helper funtion to send packets through model and format print/log prediction results
@torch.no_grad()
def _predict_print_log(features_df, *, src, dst, protocol, ttl, total_len, t_delta, tag):
    # Calls predict_df from NIDS.py
    p_arr, y_arr = predict_df(features_df, m, feature_names, scaler_mean, scaler_scale, device)

    # Returns predicitons and labels into variables
    p=float(p_arr[0])
    y=int(y_arr[0])

    # Format prediction output
    label="ATTACK DETECTED" if y==1 else "NORMAL"
    proto_txt="TCP" if int(protocol)==0 else "UDP"
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Print output to console
    print(f"{timestamp} | {tag:<7} | {label:<15} | p={p:.3f} | {src} -> {dst} | {proto_txt} | Len: {int(total_len)}")

    # Define log file columns
    if (not LOG_PATH.exists()) or LOG_PATH.stat().st_size == 0:
        with open(LOG_PATH, "w") as f:
            f.write("timestamp,label,p,src_ip,dst_ip,protocol,ttl,total_len,t_delta\n")
            
    # Write prediction to log file
    with open(LOG_PATH, "a") as f:
        f.write(f"{timestamp},{label},{p:.6f},{src},{dst},{proto_txt},{int(ttl)},{int(total_len)},{float(t_delta):.6f}\n")

# Function to process each captured packet
def process_packet(pkt):
    global last_packet_time
    current_time = pkt.time

    # Extract t_delta
    if last_packet_time == 0.0:
        t_delta = 0.0
    else:
        t_delta = float(current_time - last_packet_time)

    last_packet_time = current_time

    # Ignore non-IP packets
    if IP not in pkt: return  

    # Ignore packets without payload
    if Raw not in pkt: return  
    
    # Ignore non-TCP/UDP packets and Extract ports for filtering
    if TCP in pkt:
        protocol = 0
        dst_port = pkt[TCP].dport
        dport = dst_port
    elif UDP in pkt:
        protocol = 1
        dst_port = pkt[UDP].dport
        dport = dst_port
    else:
        return

    # Extract IP Packet header and source/destination IPs for filtering
    ip=pkt[IP]
    src_ip=str(ip.src)
    dst_ip=str(ip.dst)

    # Filter out multicast and broadcast packets
    if dst_ip.startswith(("224.","239.")) or dst_ip == "255.255.255.255":
        return
    if protocol == 1 and dport in (5353,1900,137,138):
        return

    # Extract payload length
    total_len = int(pkt[IP].len)

    # Extract TTL
    ttl = int(pkt[IP].ttl)

    # Extract Payload Bytes
    payload_bytes = []
    if Raw in pkt:
        payload_bytes = list(pkt[Raw].load)
    if len(payload_bytes) > 1500:
        payload_bytes = payload_bytes[:1500]
    else:
        padding = [0] * (1500 - len(payload_bytes))
        payload_bytes.extend(padding)

    # DataFrame creation
    row_data = payload_bytes + [ttl, total_len, protocol, t_delta]
    features = pd.DataFrame([row_data], columns=ALL_COLS)

    # Perform prediction by calling _predict_print_log() from NIDS.py
    try:
        _predict_print_log(features,
            src=str(ip.src),
            dst=str(ip.dst),
            protocol=protocol,
            ttl=ttl,
            total_len=total_len,
            t_delta=t_delta,
            tag="LIVE",
        )
    except Exception as e:
        print(f"Error during prediction: {e}")

# Helper function to inject malicious packet
@torch.no_grad()
def inject_one():
    global inj_pool

    if inj_pool is None or len(inj_pool) == 0:
        print("INJECT  | pool missing/empty")
        return
    
    # Select one random packet from injection pool
    i = int(torch.randint(low=0, high=len(inj_pool), size=(1,)).item())
    # Extract all features from selected packet
    row = inj_pool.iloc[i].copy()

    # Extract non-payload features
    ttl = int(row["ttl"])
    total_len = int(row["total_len"])
    protocol = int(row["protocol"])
    t_delta = float(row["t_delta"])

    # Create DataFrame for prediction
    features = pd.DataFrame([row.to_list()], columns=ALL_COLS)

    # Perform prediction by calling _predict_print_log() from NIDS.py 
    _predict_print_log(features, 
        src="INJECT", 
        dst="INJECT", 
        protocol=protocol, 
        ttl=ttl, 
        total_len=total_len, 
        t_delta=t_delta, 
        tag="INJECT",)

# Main function to start live detection and handle user input
def main():
    # Initialize injection pool as None
    global inj_pool
    inj_pool = None
    if os.path.exists(MAL_POOL_CSV):
        try:
            # Load injection pool from CSV
            inj_pool = pd.read_csv(MAL_POOL_CSV)

            # Ensure all required columns are present
            for c in ALL_COLS:
                if c not in inj_pool.columns:
                    inj_pool[c] = 0.0
                    
            # Keep only required columns and filter out empty payloads
            inj_pool = inj_pool[ALL_COLS]
            inj_pool = inj_pool[inj_pool[PAYLOAD_COLS].sum(axis=1) > 0].reset_index(drop=True)

            print(f"Loaded injection pool: {MAL_POOL_CSV} | rows={len(inj_pool)}")
        except Exception as e:
            inj_pool = None
            print(f"WARNING: couldn't load injection pool '{MAL_POOL_CSV}': {e} (injection disabled)")
    else:
        print(f"WARNING: injection pool not found: {MAL_POOL_CSV} (injection disabled)")

    global last_inject_press
    print("Starting live attack detection...")
    
    # This is how the live capture happens!!!
    sniffer = AsyncSniffer(prn=process_packet, store=False, filter="ip and (tcp or udp)")
    sniffer.start()

    # User input loop for packet injection and quit
    try:
        print("Controls: press 'i' to INJECT 1 sample | 'q' to quit")
        while True:
            # Check for key press
            if msvcrt.kbhit():
                ch = msvcrt.getwch().lower()
                if ch == "i":
                    # Check injection cooldown
                    now = time.time()
                    if now - last_inject_press >= INJECT_COOLDOWN:
                        last_inject_press = now
                        # Inject packet
                        inject_one()
                # Quit on 'q' key press
                elif ch == "q":
                    break
            time.sleep(0.01)
    finally:
        try:
            sniffer.stop()
        except Exception:
            pass
        print("Stopped.")

if __name__ == "__main__":
    main()