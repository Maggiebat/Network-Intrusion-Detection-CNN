from scapy.all import PcapReader, IP, TCP, UDP
import pandas as pd

# CONFIG
pcap_file = 'network_data_2days.pcap'  # <-- your file
output_csv = '2day_capture.csv'

count = 0

# Storage
rows = []

# Read packets one-by-one
print(f"Reading {pcap_file} ... (streaming mode)")
with PcapReader(pcap_file) as pcap_reader:
    for pkt in pcap_reader:
        if IP in pkt:
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            proto = pkt[IP].proto
            length = len(pkt)

            src_port = 0
            dst_port = 0

            if TCP in pkt:
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
            elif UDP in pkt:
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport

            rows.append({
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'protocol': proto,
                'length': length
            })
        print(f"\rProcessed {count} packets", end="")
        count += 1

# Save to CSV
df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)
print(f"Done! Extracted {len(df)} packets to {output_csv}")