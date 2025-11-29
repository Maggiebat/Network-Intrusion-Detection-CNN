from scapy.all import sniff, IP, TCP, UDP, Raw
import pandas as pd

last_packet_time = 0.0

PAYLOAD_COLS = [f'payload_byte_{i+1}' for i in range(1500)]
ALL_COLS = ['ttl', 'total_len', 'proto_bin', 't_delta'] + PAYLOAD_COLS

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
    row_data = [ttl, total_len, proto_bin, t_delta] + payload_bytes
    features = pd.DataFrame([row_data], columns=ALL_COLS)

    print(f"{ttl} | {total_len} | {'TCP' if proto_bin==0 else 'UDP'} | {t_delta} | {payload_bytes[:10]}...") 

    with open("train_data_log.csv", "a") as f:
        f.write(f"{ttl} | {total_len} | {'TCP' if proto_bin==0 else 'UDP'} | {t_delta} | {payload_bytes[:1500]}...\n")

print("Starting live attack detection... Press Ctrl+C to stop.")
# this is how the live capture happens!!!
sniff(prn=process_packet, store=False)