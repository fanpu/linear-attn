#!/bin/bash
# usage: submit.sh <time> <mem> <train.py args...>   (one pasar job per run)
T=$1; M=$2; shift 2
NAME=$(echo "$@" | sed -E 's/--task //; s/--arch //; s/--opt //; s/--labels //; s/--seed /s/; s/ --/ /g; s/ +/_/g')
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True pasar submit --json --time "$T" --mem "$M" --name "or_$NAME" \
  --tag art-one-road --by art-oneroad --note "One Road atlas: function-space training trajectory ($*)" \
  --cwd /home/fzeng/ml/research/art/one-road -- ../.venv/bin/python train.py "$@" | python3 -c "import sys,json; j=json.load(sys.stdin); print(j['id'], j['name'], j['state'])"
