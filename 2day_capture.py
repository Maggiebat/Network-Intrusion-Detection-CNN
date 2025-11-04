from scapy.all import sniff, PcapWriter
import csv
from datetime import datetime

# Config
capture_duration = 2 * 24 * 60 * 60  # 2 days
interface = "enp5s0"
csv_file = "network_data_2days.csv"
pcap_file = "network_data_2days.pcap"

# Setup CSV
csv_fields = ["timestamp", "src_ip", "dst_ip", "protocol", "length"]
csv_file_handle = open(csv_file, mode='w', newline='')
csv_writer = csv.DictWriter(csv_file_handle, fieldnames=csv_fields)
csv_writer.writeheader()

# Setup PCAP writer (streaming mode)
pcap_writer = PcapWriter(pcap_file, append=True, sync=True)

# Process packets one-by-one
def process_packet(pkt):
    if pkt.haslayer("IP"):
        row = {
            "timestamp": datetime.now().isoformat(),
            "src_ip": pkt["IP"].src,
            "dst_ip": pkt["IP"].dst,
            "protocol": pkt["IP"].proto,
            "length": len(pkt)
        }
        csv_writer.writerow(row)
        csv_file_handle.flush()
        pcap_writer.write(pkt)

print(f"[*] Capturing traffic on {interface} for 2 days...")
sniff(iface=interface, prn=process_packet, timeout=capture_duration, store=False)

csv_file_handle.close()
pcap_writer.close()
print("[+] Capture complete.")

