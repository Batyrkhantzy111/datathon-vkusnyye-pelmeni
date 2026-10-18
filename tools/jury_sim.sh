#!/usr/bin/env bash
# Jury simulation: upload N videos at once through the public URL (spec §10 flow) and poll each job until done.
# Prints per-job wall time from upload to result, verdict and violation types. Run from WSL:
#   bash tools/jury_sim.sh KEY URL file1.mov file2.mov ...
set -u
KEY="$1"; URL="$2"; shift 2
tmp=$(mktemp -d)
start=$(date +%s)
i=0
for f in "$@"; do
  i=$((i + 1))
  (
    t0=$(date +%s)
    id=""
    for attempt in 1 2 3; do
      j=$(curl -s -m 180 -X POST "$URL/api/v1/analyze" -H "X-API-Key: $KEY" -F "file=@$f")
      id=$(echo "$j" | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])' 2>/dev/null) && break
      sleep 2
    done
    [ -z "$id" ] && { echo "$(basename "$f") upload failed: $j" > "$tmp/$i"; exit 0; }
    while :; do
      s=$(curl -s -m 30 "$URL/api/v1/jobs/$id" -H "X-API-Key: $KEY")
      st=$(echo "$s" | python3 -c 'import sys,json;print(json.load(sys.stdin)["status"])' 2>/dev/null)
      [ "$st" = "done" ] || [ "$st" = "failed" ] && break
      [ $(( $(date +%s) - t0 )) -gt 600 ] && { st="timeout"; break; }
      sleep 4
    done
    echo "$s" | python3 -c '
import sys, json
d = json.load(sys.stdin)
v = [(x["type_id"], x["timecode_sec"]) for x in d.get("violations", [])]
print(d.get("status"), d.get("verdict"), v, "server", d.get("processing_sec"))' > "$tmp/$i.r" 2>/dev/null
    echo "$(basename "$f") $id wall=$(( $(date +%s) - t0 ))s $(cat "$tmp/$i.r" 2>/dev/null || echo "$st")" > "$tmp/$i"
  ) &
done
wait
cat "$tmp"/[0-9]*
echo "all done in $(( $(date +%s) - start ))s"
rm -rf "$tmp"
