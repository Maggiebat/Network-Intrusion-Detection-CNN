# collect_data.py  (training-csv capture; CICIDS-style cols + payload features; TCP/UDP only; no empty payloads)
import csv, argparse
from pathlib import Path
from datetime import datetime
from scapy.all import sniff, IP, TCP, UDP, Raw

PAYLOAD_N=1500
PAYLOAD_COLS=[f"payload_byte_{i+1}" for i in range(PAYLOAD_N)]
FEATURE_COLS=PAYLOAD_COLS+["ttl","total_len","protocol","t_delta"]
META_COLS=["Flow ID","Source IP","Source Port","Destination IP","Destination Port","Timestamp"]
ALL_COLS=META_COLS+FEATURE_COLS+["Label"]

last_packet_time=0.0

def ensure_header(path:Path):
    if (not path.exists()) or path.stat().st_size==0:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("w",newline="") as f: csv.writer(f).writerow(ALL_COLS)

def flow_id(src_ip,dst_ip,sport,dport,protocol):
    proto="TCP" if protocol==0 else "UDP"
    return f"{src_ip}-{dst_ip}-{sport}-{dport}-{proto}"

def process_packet(pkt,*,out_path:Path,label:str,require_payload=True,drop_multicast=True,drop_local_chatter=True):
    global last_packet_time
    if IP not in pkt: return

    ip=pkt[IP]
    if TCP in pkt: l4=pkt[TCP]; protocol=0
    elif UDP in pkt: l4=pkt[UDP]; protocol=1
    else: return  # not used in your dataset/live model

    src_ip=str(ip.src); dst_ip=str(ip.dst); sport=int(l4.sport); dport=int(l4.dport)

    # drop packets your live capture doesn’t "use" (noise): multicast/broadcast + common local discovery chatter
    if drop_multicast:
        if dst_ip.startswith("224.") or dst_ip.startswith("239.") or dst_ip=="255.255.255.255": return
    if drop_local_chatter and protocol==1 and dport in (5353,1900,137,138): return  # mDNS/SSDP/NetBIOS

    # skip empty payloads (TCP handshakes/ACKs etc.)
    if require_payload:
        if Raw not in pkt: return
        raw=bytes(pkt[Raw].load)
        if len(raw)==0: return
    else:
        raw=bytes(pkt[Raw].load) if Raw in pkt else b""

    now=float(pkt.time); t_delta=0.0 if last_packet_time==0.0 else (now-last_packet_time); last_packet_time=now
    ttl=int(ip.ttl); total_len=int(ip.len)

    payload=list(raw[:PAYLOAD_N]); payload += [0]*max(0, PAYLOAD_N-len(payload))
    ts=datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    row_m=[flow_id(src_ip,dst_ip,sport,dport,protocol), src_ip, sport, dst_ip, dport, ts]
    row_f=payload+[ttl,total_len,protocol,float(t_delta)]
    with out_path.open("a",newline="") as f: csv.writer(f).writerow(row_m+row_f+[label])

    print(f"{ts} | saved | {label} | {src_ip}:{sport}->{dst_ip}:{dport} | {'TCP' if protocol==0 else 'UDP'} | iplen={total_len} | tΔ={t_delta:.6f} | pay={len(raw)}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="CICIDS_live_capture_preprocessed.csv")
    ap.add_argument("--label",default="BENIGN")                  # set to ATTACK when capturing malicious traffic
    ap.add_argument("--iface",default=None)                      # optional on Windows if scapy picks wrong adapter
    ap.add_argument("--no-require-payload",action="store_true")  # include empty payload packets (NOT recommended)
    args=ap.parse_args()

    out_path=Path(args.out)
    ensure_header(out_path)
    print("Writing:",out_path.resolve())
    print("Filter: ip and (tcp or udp) and not multicast and not broadcast; plus skip empty payloads; plus drop mDNS/SSDP/NetBIOS UDP noise")
    print("Label:",args.label,"| iface:",args.iface or "(auto)")

    sniff(
        prn=lambda p: process_packet(p,out_path=out_path,label=args.label,require_payload=(not args.no_require_payload)),
        store=False,
        iface=args.iface,
        filter="ip and (tcp or udp)"
    )

if __name__=="__main__": main()
