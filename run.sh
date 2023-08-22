#!/bin/bash

declare -a pids=()

for f in ./movs/*; do
  echo $f
  python3 process_video.py "$f" &
  pids+=($!)
done

for pid in ${pids[*]}; do
  wait $pid
done

