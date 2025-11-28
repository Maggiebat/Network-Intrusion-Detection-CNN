from scapy.all import sniff, IP, TCP, UDP
import pandas as pd
import pickle
import ipaddress
from datetime import datetime

#  Load trained model **change to CNN model**
with open('ids.pkl', 'rb') as f:
    model = pickle.load(f)

# converts IP to integer value
def ip_to_int(ip_address):
    try:
        return int(ipaddress.ip_address(ip_address))
    except ValueError:
        return 0  # fallback

# begins packet processing
def process_packet(pkt):
    if IP in pkt:
        src_ip = ip_to_int(pkt[IP].src)
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

        # Prepare features for prediction **need to alter to match our features**
        features = pd.DataFrame([{
            'protocol': proto,
        }])

        # Perform prediction **see note at top and adjust as needed**
        prediction = model.predict(features)[0]
        score = model.decision_function(features)[0]

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        label = 'ATTACK DETECTED' if prediction == -1 else 'NORMAL'

        # Print alert
        print(f"{timestamp} | {label} | {pkt[IP].src} → {pkt[IP].dst} | proto={proto}, len={length} | score={score:.4f}")

        # Optional logging of attacks
        with open("live_attack_log.csv", "a") as f:
            f.write(f"{timestamp},{pkt[IP].src},{pkt[IP].dst},{src_port},{dst_port},{proto},{length},{score:.4f},{label}\n")

# Opens up interface for sniffing
print("Starting live attack detection... Press Ctrl+C to stop.")
sniff(iface="enp5s0", prn=process_packet, store=False)
