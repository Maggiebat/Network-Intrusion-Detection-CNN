#!/usr/bin/env bash
# start.sh — create folders + download required Google Drive files into /data and /model
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p data model results

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python - <<'PY'
import os
from pathlib import Path
import gdown

root = Path(os.getcwd())
data = root / "data"
model = root / "model"
data.mkdir(exist_ok=True)
model.mkdir(exist_ok=True)

files = [
    ("102qs3YTnn8_rAcya4FQkIAKsRv_6w1Gn", data / "malicious_packets.csv"),
    ("1aQRpdTw6PhMNGKlPixJ0YI3gReGgmrm6", model / "NIDS.pt"),
]

for fid, out in files:
    if out.exists() and out.stat().st_size > 0:
        print(f"ok: {out}")
        continue
    url = f"https://drive.google.com/uc?id={fid}"
    print(f"download: {out.name}")
    gdown.download(url, str(out), quiet=False)

print("done.")
PY