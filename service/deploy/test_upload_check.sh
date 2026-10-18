#!/usr/bin/env bash
# Upload-validation check against a running instance: every open clip must get a job_id (200),
# a clip whose media data is zeroed out (container headers intact) must get 400.
# Usage: bash test_upload_check.sh [base_url] [api_key]
set -u
BASE="${1:-http://127.0.0.1:8010}"
KEY="${2:-testkey}"
ok=0; bad=0
for f in ~/forte/videos/X-*.mov; do
  t0=$(date +%s.%N)
  code=$(curl -s -o /tmp/upl.json -w '%{http_code}' -H "X-API-Key: $KEY" -F "file=@$f" "$BASE/api/v1/analyze")
  dt=$(echo "$(date +%s.%N) - $t0" | bc)
  if [ "$code" = 200 ]; then ok=$((ok+1)); else bad=$((bad+1)); echo "REJECTED $f $code $(cat /tmp/upl.json)"; fi
  printf '%s %s %.2fs\n' "$(basename "$f")" "$code" "$dt"
done
echo "accepted=$ok rejected=$bad"

# Damaged copy: keep the first 64 KB and last 256 KB, zero everything between (mdat payload gone, moov intact).
src=~/forte/videos/X-02.mov
dmg=/tmp/damaged.mov
cp "$src" "$dmg"
size=$(stat -c %s "$dmg")
python3 - "$dmg" "$size" <<'EOF'
import sys
p, size = sys.argv[1], int(sys.argv[2])
with open(p, "r+b") as f:
    f.seek(65536)
    f.write(b"\0" * (size - 65536 - 262144))
EOF
echo "ffprobe on damaged copy:"; ffprobe -v error -show_entries stream=codec_name,width,height,duration -of csv "$dmg"
echo "upload damaged copy:"
curl -s -w ' %{http_code}\n' -H "X-API-Key: $KEY" -F "file=@$dmg;filename=damaged.mov" "$BASE/api/v1/analyze"
