from scapy.all import PcapReader, IP, TCP, UDP, Raw
import pandas as pd
import time

# collect info from each packet
# format into a datafram
    # payload byte 1, payload byte 2, ..., payload byte 1500, ttl, total length, protocol, t delta

class Packet_Processor:
    pcap_file = ''
    output_csv = '' 

    def __init__(self):
        self.rows = []
        self.last_packet_time = None
        self.payload_cols = [f'payload_byte_{i}' for i in range(1, 1501)]
        self.other_cols = ['ttl', 'total_len', 'protocol', 't_delta']
        self.columns = self.payload_cols + self.other_cols

    def process_packets(self, pkt):
        current_time = time.time()
        if self.last_packet_time is None:
            t_delta = 0
        else:
            t_delta = float(current_time - self.last_packet_time)
        
        self.last_packet_time = current_time

        if not pkt.haslayer(IP):
            return  # Skip non-IP packets
        
        protocol = pkt[IP].proto
        
        if protocol not in [6, 17]:  # TCP=6, UDP=17
            return  # Skip non-TCP/UDP packets
        
        ttl = pkt[IP].ttl
        total_len = protocol.len

        if pkt.haslayer(Raw):
            raw_bytes = bytes(pkt[Raw])
        else:
            raw_bytes = b''
        
        byte_list = list(raw_bytes)

        if len(byte_list) < 1500:
            byte_list = byte_list[:1500]


