from scapy.all import sniff, IP, TCP, UDP, Raw
import pandas as pd
import csv, argparse
from pathlib import Path
from datetime import datetime
import torch

from NIDS import load_nids, predict_df

# Configure paths
ROOT=Path(__file__).resolve().parent
DATA_DIR=ROOT/"data"; DATA_DIR.mkdir(parents=True,exist_ok=True)
MODEL_DIR=ROOT/"model"; MODEL_DIR.mkdir(parents=True,exist_ok=True)
CKPT_PATH=MODEL_DIR/"NIDS.pt"

# Intiialize t_delta to 0
last_packet_time = 0.0

# Define dataframe columns
PAYLOAD_COLS = [f'payload_byte_{i+1}' for i in range(1500)]
ALL_COLS = PAYLOAD_COLS + ['ttl', 'total_len', 'protocol', 't_delta']

# Function to process each captured packet
def process_packet(pkt, *, out_path, mode, device, model_pack):
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

    # payload bytes extraction
    payload_bytes = []
    if Raw in pkt:
        payload_bytes = list(pkt[Raw].load)
    if len(payload_bytes) > 1500:
        payload_bytes = payload_bytes[:1500]
    else:
        padding = [0] * (1500 - len(payload_bytes))
        payload_bytes.extend(padding)

    # Initialize DataFrame
    row_data = payload_bytes + [ttl, total_len, protocol, t_delta]
    features = pd.DataFrame([row_data], columns=ALL_COLS)
    label_txt="UNKNOWN"; p=None; y=None

    # Determine label based on mode
    if mode == "benign":
        y = 0; label_txt = "BENIGN"
    elif mode == "malicious":
        y = 1; label_txt = "MALICIOUS"
    elif mode == "predict":
        m, feature_names, scaler_mean, scaler_scale = model_pack
        p_arr, y_arr = predict_df(features, m, feature_names, scaler_mean, scaler_scale, device)
        p = float(p_arr[0])
        y = int(y_arr[0])
        label_txt = "MALICIOUS" if y == 1 else "BENIGN"

    # Save to CSV
    row = row_data if y is None else (row_data + [y])
    with Path(out_path).open("a", newline="") as f:
        csv.writer(f).writerow(row)

    # Format collection output
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    proto_txt = 'TCP' if protocol == 0 else 'UDP'

    # Print output to console
    if p is None:
        print(f"{timestamp} | SAVED | {label_txt} | {src_ip} -> {dst_ip} | {proto_txt} | Len: {total_len}")
    else:
        print(f"{timestamp} | SAVED | {label_txt} | p={p:.3f} | {src_ip} -> {dst_ip} | {proto_txt} | Len: {total_len}")

# Main function to start live detection and handle arguments
def main():
    ap=argparse.ArgumentParser()

    # Default mode collects all packets without labels
    g=ap.add_mutually_exclusive_group()

    # Benign mode lablels all collected packets as benign
    g.add_argument("-benign", action="store_true")

    # Malicious mode labels all collected packets as malicious
    g.add_argument("-malicious", action="store_true")

    # Predict mode uses model predicitons to label each packet
    g.add_argument("-predict", action="store_true")

    args=ap.parse_args()

    mode = "predict" if args.predict else ("benign" if args.benign else ("malicious" if args.malicious else "default"))
    out_name = {"default":"packet_data.csv","benign":"benign_packet_data.csv","malicious":"malicious_packet_data.csv","predict":"predicted_packet_data.csv"}[mode]

    # Define output path based on mode
    out_path = DATA_DIR/out_name

    # Create output CSV file with colums if it doesn't exist
    cols = ALL_COLS if mode == "default" else ALL_COLS + ["label"]
    out_path=Path(out_path)
    if (not out_path.exists()) or out_path.stat().st_size == 0:
        out_path.parent.mkdir(parents=True,exist_ok=True)
        with out_path.open("w", newline="") as f:
            csv.writer(f).writerow(cols)

    # Initialize model
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_pack=None

    if mode == "predict":
        try:
            # Load model using load_nids() from NIDS.py if in predict mode
            m, val_best_f1, feature_names, scaler_mean, scaler_scale = load_nids(CKPT_PATH, device=device)
            m.eval()
            model_pack = (m, feature_names, scaler_mean, scaler_scale)
        except FileNotFoundError:
            print("Model file 'model/NIDS.pt' not found. Please train first, or place NIDS.pt in the model folder.")
            return
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    print("Starting live attack detection... Press Ctrl+C to stop.")

    # This is how the live capture happens!!!
    sniff(prn=lambda p: process_packet(p, out_path=out_path, mode=mode, device=device, model_pack=model_pack), store=False, filter="ip and (tcp or udp)")

if __name__ == "__main__":
    main()