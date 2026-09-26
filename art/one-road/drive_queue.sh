#!/bin/bash
# Feed a list of train.py configs to pasar, keeping at most $2 of this agent's jobs active at once.
Q=$1; MAX=${2:-4}
cd /home/fzeng/ml/research/art/one-road
while IFS= read -r line; do
  [ -z "$line" ] && continue
  while true; do
    n=$(pasar ls --json | python3 -c "import sys,json; j=json.load(sys.stdin); print(sum(1 for x in j if x.get('submitter')=='art-oneroad' and x['state'] in ('running','queued')))")
    [ "$n" -lt "$MAX" ] && break
    sleep 10
  done
  ./submit.sh $line
  sleep 2
done < "$Q"
echo DONE
