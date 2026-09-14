#!/usr/bin/env bash
# Sequential measurement campaign (idle GPU). Before each job: wait until GPU temp <= 50 C and util == 0.
set -e
cd /home/fzeng/ml/research/hardware/lattice
P=/home/fzeng/ml/research/art/.venv/bin/python
for job in "$@"; do
  for i in $(seq 1 60); do
    read -r u t <<<"$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader,nounits | tr -d ',')"
    [ "$t" -le 50 ] && [ "$u" -le 5 ] && break; sleep 5
  done
  log="logs/$(echo $job | tr ' -' '_' | tr -s '_').log"
  echo "=== $(date -Is) $job (start temp ${t}C util ${u}%)" >> logs/campaign.log
  $P sweep.py $job 2>&1 | grep --line-buffered -v USDT > "$log"
  tail -1 "$log" | cut -c1-300 >> logs/campaign.log
done
echo "=== $(date -Is) done" >> logs/campaign.log
