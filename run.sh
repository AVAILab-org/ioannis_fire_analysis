#!/bin/bash

declare -a pids=()

for f in ./movs/*; do
  echo $f
  python3 main.py "$f" &
  pids+=($!)
done

for pid in ${pids[*]}; do
  wait $pid
done

