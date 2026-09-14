#!/usr/bin/env bash
# Staircase campaign: random pointer chase on one core of each type/cluster, 4 KB .. 2 GB, 4K vs THP pages,
# plus the sequential null and heartbeats. Run on an idle machine (GPU idle too: memory is unified).
# Waits for GPU util <= 5 % before starting. ~40 min total.
set -u
cd "$(dirname "$0")"
mkdir -p cache logs
while [ "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)" -gt 5 ]; do sleep 10; done
echo "start $(date)" > logs/campaign.log
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv >> logs/campaign.log
cat /proc/loadavg >> logs/campaign.log
for cpu in 0 5 10 15; do
  for huge in 0 1; do
    tag="c${cpu}_huge${huge}"
    h=""; [ $huge = 1 ] && h="--huge"
    echo "$(date) $tag" >> logs/campaign.log
    ./chase --cpu $cpu $h --min 1024 --max 1073741824 --per-octave 8 --loads 20000000 --reps 5 > cache/stair_${tag}.tsv 2>> logs/campaign.log
  done
  echo "$(date) c${cpu} seq" >> logs/campaign.log
  ./chase --cpu $cpu --seq --min 1024 --max 1073741824 --per-octave 4 --loads 20000000 --reps 3 > cache/stair_c${cpu}_seq.tsv 2>> logs/campaign.log
done
# node-size diagnostic (8-byte nodes: 8x more chain entries per byte; separates cache capacity from prefetcher history)
for cpu in 0 5; do
  echo "$(date) c${cpu} node8" >> logs/campaign.log
  ./chase --cpu $cpu --node 8 --min 1024 --max 268435456 --per-octave 8 --loads 20000000 --reps 5 > cache/stair_c${cpu}_node8.tsv 2>> logs/campaign.log
done
# heartbeats: 32 KB (L1-resident) and 64 MB (DRAM) working sets, 1e6 chunks of 2000 loads
for cpu in 0 5; do
  ./chase --cpu $cpu --heartbeat 32768 --samples 1000000 --chunk 2000 > cache/hb_c${cpu}_l1.bin 2>> logs/campaign.log
  ./chase --cpu $cpu --heartbeat 67108864 --samples 300000 --chunk 2000 > cache/hb_c${cpu}_dram.bin 2>> logs/campaign.log
done
echo "done $(date)" >> logs/campaign.log
