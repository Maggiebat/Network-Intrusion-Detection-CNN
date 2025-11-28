from scapy.all import sniff, IP, TCP, UDP, Raw
import pandas as pd
import pickle
import ipaddress
from datetime import datetime
import torch

#  Load trained model **change to CNN model**
with open('ids_cnn.pkl', 'rb') as f:
    model = pickle.load(f)

# begins packet processing
def process_packet(pkt):
    if IP in pkt:
        proto = pkt[IP].proto
        length = len(pkt)

        # Initialize ports
        # src_port = 0
        # dst_port = 0

        # Extract ports if TCP or UDP
        if TCP in pkt:
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
        elif UDP in pkt:
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport

        # Can I add the t_delta (time between packets) feature here?

        # payload bytes extraction added here

        # Prepare features for prediction **need to alter to match our features**
        features = pd.DataFrame([{
            'protocol': proto,
            'total_len': length,
            't_delta': 0  # Placeholder for time delta feature
        }])

        # Perform prediction **see note at top and adjust as needed**
        prediction = model.predict(features)[0]

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        label = 'ATTACK DETECTED' if prediction == -1 else 'NORMAL'

        # Print alert
        print(f"{timestamp} | {label} | {pkt[IP].src} → {pkt[IP].dst} | proto={proto}, len={length}")

        # Optional logging of attacks
        with open("live_attack_log.csv", "a") as f:
            f.write(f"{timestamp},{pkt[IP].src},{pkt[IP].dst},{src_port},{dst_port},{proto},{length},{label}\n")
            
# Opens up interface for sniffing
print("Starting live attack detection... Press Ctrl+C to stop.")
sniff(iface="enp5s0", prn=process_packet, store=False)
