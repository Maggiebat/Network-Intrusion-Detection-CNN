from scapy.all import sniff, IP, TCP, UDP, Raw
import pandas as pd
from datetime import datetime
import torch
from model import FC_CNN

#  Load trained model
try:
    with open('cicids_cnn_best_ckpt.pt', 'rb') as f:
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt=torch.load("cicids_cnn_best_ckpt.pt",map_location=device)

        m=FC_CNN(num_features=int(ckpt["num_features"]), cfg=ckpt["cfg"]).to(device)
        m.load_state_dict(ckpt["model_state"])
    m.eval()
except FileNotFoundError:
    print("Model file 'cicids_cnn_best_ckpt.pt' not found. Please ensure the model is in the current directory.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

last_packet_time = 0.0

PAYLOAD_COLS = [f'payload_byte_{i+1}' for i in range(1500)]
ALL_COLS = PAYLOAD_COLS + ['ttl', 'total_len', 'proto_bin', 't_delta']

# begins packet processing
def process_packet(pkt):
    global last_packet_time
    current_time = pkt.time

    # t_delta extraction
    if last_packet_time == 0.0:
        t_delta = 0.0
    else:
        t_delta = float(current_time - last_packet_time)

    last_packet_time = current_time

    if IP not in pkt:
        return  # Ignore non-IP packets

    # Extract ports if TCP or UDP
    if TCP in pkt:
        proto_bin = 0
        src_port = pkt[TCP].sport
        dst_port = pkt[TCP].dport
    elif UDP in pkt:
        proto_bin = 1
        src_port = pkt[UDP].sport
        dst_port = pkt[UDP].dport
    else: return  # Ignore non-TCP/UDP packets

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

    # DataFrame creation
    row_data = payload_bytes + [ttl, total_len, proto_bin, t_delta]
    features = pd.DataFrame([row_data], columns=ALL_COLS)

    # Perform prediction **see note at top and adjust as needed**
    try:
        input_tensor = torch.tensor(features.values, dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            output = m(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
        
        label = 'ATTACK DETECTED' if prediction == 1 else 'NORMAL'
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"{label} | {ttl} | {total_len} | {'TCP' if proto_bin==0 else 'UDP'} | {t_delta} | {payload_bytes[:10]}...") 

        with open("NIDS_results_log.csv", "a") as f:
            f.write(f"{label} | {ttl} | {total_len} | {'TCP' if proto_bin==0 else 'UDP'} | {t_delta} | {payload_bytes[:10]}...\n")

    except Exception as e:
        print(f"Error during prediction: {e}")
            
# Opens up interface for sniffing
print("Starting live attack detection... Press Ctrl+C to stop.")
# this is how the live capture happens!!!
sniff(prn=process_packet, store=False)
